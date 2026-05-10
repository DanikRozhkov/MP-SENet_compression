import torch
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.model import phase_losses
from dataset import Dataset, get_dataset_filelist, mag_pha_stft, mag_pha_istft

def fine_tune_model(model, device, checkpoint_path, h, num_files=1000,
                    batch_size=2, epochs=3, lr=1e-5, seed=1234):
    """
    Дообучение модели после прунинга
    model: модель
    device: 'cpu' или 'cuda' / 'mps'
    checkpoint_path: папка, куда сохранять временные чекпоинты (опционально)
    h: конфиг модели
    num_files: количество обучающих файлов для fine-tuning (берутся из training.txt)
    batch_size: 2
    epochs: 3-5
    lr: 1e-5
    """
    random.seed(seed)
    
    class Args:
        input_training_file = "VoiceBank+DEMAND/training.txt"
        input_validation_file = "VoiceBank+DEMAND/test.txt"
        input_clean_wavs_dir = "VoiceBank+DEMAND/wav_clean"
        input_noisy_wavs_dir = "VoiceBank+DEMAND/wav_noisy"
    args = Args()
    
    all_train_indexes, _ = get_dataset_filelist(args)
    train_indexes = random.sample(all_train_indexes, num_files)
    print(f"Fine-tuning on {len(train_indexes)} files")

    dataset = Dataset(
        training_indexes=train_indexes,
        clean_wavs_dir=args.input_clean_wavs_dir,
        noisy_wavs_dir=args.input_noisy_wavs_dir,
        segment_size=h.segment_size,
        sampling_rate=h.sampling_rate,
        split=True,
        shuffle=True,
        n_cache_reuse=0,
        device=device
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (clean_audio, noisy_audio) in enumerate(dataloader):
            if batch_idx % 10 == 0:
                print(batch_idx)
            clean_audio = clean_audio.to(device)
            noisy_audio = noisy_audio.to(device)

            clean_mag, clean_pha, clean_com = mag_pha_stft(
                clean_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )
            noisy_mag, noisy_pha, _ = mag_pha_stft(
                noisy_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )

            mag_g, pha_g, com_g = model(noisy_mag, noisy_pha)

            audio_g = mag_pha_istft(
                mag_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )
            mag_g_hat, pha_g_hat, com_g_hat = mag_pha_stft(
                audio_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )

            loss_mag = F.mse_loss(clean_mag, mag_g)

            loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, pha_g)
            loss_pha = loss_ip + loss_gd + loss_iaf
            loss_com = F.mse_loss(clean_com, com_g) * 2
            loss_stft = F.mse_loss(com_g, com_g_hat) * 2
            loss_time = F.l1_loss(clean_audio, audio_g)

            loss = loss_mag * 0.9 + loss_pha * 0.3 + loss_com * 0.1 + loss_stft * 0.1 + loss_time * 0.2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Fine-Tune Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")

    model.eval()
    return model