import torch
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.model import MPNet, phase_losses
from dataset import Dataset, get_dataset_filelist, mag_pha_stft, mag_pha_istft
from env import AttrDict
import json
import os

def save_checkpoint(state, checkpoint_dir, filename):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(checkpoint_dir, filename, map_location='cpu'):
    path = os.path.join(checkpoint_dir, filename)
    if not os.path.exists(path):
        return None
    print(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=map_location)
    return checkpoint

def distill_model(model, config, device):
    comp = config.get('compression', {})
    teacher_ckpt = comp.get('teacher_checkpoint')
    if teacher_ckpt is None:
        teacher_ckpt = config['data']['checkpoint_file']
    
    student_save_path = comp.get('save_path', 'student_model.pth')
    student_params = comp.get('student_params', {})
    dense_channel = student_params.get('dense_channel', 32)
    num_tsconformers = student_params.get('num_tsconformers', 2)
    num_files = comp.get('num_files', 1000)
    batch_size = comp.get('batch_size', 2)
    total_epochs = comp.get('epochs', 10)
    lr = comp.get('lr', 1e-4)
    alpha = comp.get('alpha', 0.7)
    seed = comp.get('seed', 1234)
    resume_dir = comp.get('resume_checkpoint_dir', None)
    save_dir = comp.get('save_checkpoint_dir', '/kaggle/working/checkpoints')
    checkpoint_file = comp.get('checkpoint_filename', 'distill_checkpoint.pt')

    random.seed(seed)
    torch.manual_seed(seed)

    teacher_config_path = os.path.join(os.path.split(teacher_ckpt)[0], 'config.json')
    with open(teacher_config_path, 'r') as f:
        teacher_config_dict = json.load(f)
    h_teacher = AttrDict(teacher_config_dict)

    # Учитель
    teacher = MPNet(h_teacher).to(device)
    checkpoint = torch.load(teacher_ckpt, map_location='cpu')
    teacher.load_state_dict(checkpoint['generator'])
    teacher.eval()
    print(f"Teacher loaded from {teacher_ckpt}")

    # Студент
    student_config_dict = teacher_config_dict.copy()
    student_config_dict['dense_channel'] = dense_channel
    student_config_dict['num_tsconformers'] = num_tsconformers
    h_student = AttrDict(student_config_dict)
    student = MPNet(h_student).to(device)
    student.train()
    print(f"Student created: dense_channel={dense_channel}, num_tsconformers={num_tsconformers}")

    class Args:
        input_training_file = "VoiceBank+DEMAND/training.txt"
        input_validation_file = "VoiceBank+DEMAND/test.txt"
        input_clean_wavs_dir = "VoiceBank+DEMAND/wav_clean"
        input_noisy_wavs_dir = "VoiceBank+DEMAND/wav_noisy"
    args = Args()
    all_train_indexes, _ = get_dataset_filelist(args)
    # train_indexes = random.sample(all_train_indexes, min(num_files, len(all_train_indexes)))
    print(f"Distillation on {len(all_train_indexes)} files")

    dataset = Dataset(
        training_indexes=all_train_indexes,
        clean_wavs_dir=args.input_clean_wavs_dir,
        noisy_wavs_dir=args.input_noisy_wavs_dir,
        segment_size=h_student.segment_size,
        sampling_rate=h_student.sampling_rate,
        split=True,
        shuffle=True,
        n_cache_reuse=0,
        device=device
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(student.parameters(), lr=lr)

    # Возобновление из чекпоинта
    start_epoch = 0
    if resume_dir is not None:
        ckpt = load_checkpoint(resume_dir, checkpoint_file, map_location='cpu')
        if ckpt is not None:
            student.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1
            if 'random_state' in ckpt:
                random.setstate(ckpt['random_state'])
                torch.set_rng_state(ckpt['torch_rng_state'])
            print(f"Resuming from epoch {start_epoch}")
        else:
            print("No checkpoint found, starting from scratch.")

    for epoch in range(start_epoch, total_epochs):
        total_loss = 0.0
        student.train()
        for batch_idx, (clean_audio, noisy_audio) in enumerate(dataloader):
            clean_audio = clean_audio.to(device)
            noisy_audio = noisy_audio.to(device)

            clean_mag, clean_pha, clean_com = mag_pha_stft(
                clean_audio, h_student.n_fft, h_student.hop_size, h_student.win_size, h_student.compress_factor
            )
            noisy_mag, noisy_pha, _ = mag_pha_stft(
                noisy_audio, h_student.n_fft, h_student.hop_size, h_student.win_size, h_student.compress_factor
            )

            with torch.no_grad():
                teacher_mag, teacher_pha, _ = teacher(noisy_mag, noisy_pha)

            student_mag, student_pha, student_com = student(noisy_mag, noisy_pha)

            audio_g = mag_pha_istft(
                student_mag, student_pha, h_student.n_fft, h_student.hop_size, h_student.win_size, h_student.compress_factor
            )
            mag_g_hat, _, com_g_hat = mag_pha_stft(
                audio_g, h_student.n_fft, h_student.hop_size, h_student.win_size, h_student.compress_factor
            )
            loss_mag = F.mse_loss(clean_mag, student_mag)
            loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, student_pha)
            loss_pha = loss_ip + loss_gd + loss_iaf
            loss_com = F.mse_loss(clean_com, student_com) * 2
            loss_stft = F.mse_loss(student_com, com_g_hat) * 2
            loss_time = F.l1_loss(clean_audio, audio_g)
            loss_standard = 0.9*loss_mag + 0.3*loss_pha + 0.1*loss_com + 0.1*loss_stft + 0.2*loss_time

            loss_distill = F.mse_loss(student_mag, teacher_mag) + F.mse_loss(student_pha, teacher_pha)

            loss = alpha * loss_distill + (1 - alpha) * loss_standard

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{total_epochs}, Loss: {avg_loss:.6f}")

        ckpt_state = {
            'epoch': epoch,
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'random_state': random.getstate(),
            'torch_rng_state': torch.get_rng_state(),
        }
        save_checkpoint(ckpt_state, save_dir, checkpoint_file)

    torch.save({'generator': student.state_dict()}, "/kaggle/working/student_model.pth")
    print(f"Студент сохранён в {student_save_path}")
    return student