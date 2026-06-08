from compression.prune import compute_global_sparsity
import torch
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
import nncf
from nncf import NNCFConfig
from nncf.torch import create_compressed_model
from models.model import phase_losses
from dataset import Dataset, get_dataset_filelist, mag_pha_stft, mag_pha_istft
from compression.utils import get_model_size_mb

def fine_tune_with_nncf_pruning(
    model,
    device,
    h,
    num_files=6,
    batch_size=2,
    epochs=15,
    lr=1e-5,
    seed=1234,
    pruning_target=0.5,
    pruning_steps=10,
    pruning_frequency=2,
    num_init_steps=2,
    filter_importance_type="L2",
):
    random.seed(seed)
    torch.manual_seed(seed)

    class Args:
        input_training_file = "VoiceBank+DEMAND/training.txt"
        input_validation_file = "VoiceBank+DEMAND/test.txt"
        input_clean_wavs_dir = "VoiceBank+DEMAND/wav_clean"
        input_noisy_wavs_dir = "VoiceBank+DEMAND/wav_noisy"
    args = Args()

    all_train_indexes, _ = get_dataset_filelist(args)
    train_indexes = random.sample(all_train_indexes, min(num_files, len(all_train_indexes)))
    print(f"Fine-tuning with NNCF pruning on {len(train_indexes)} files")

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

    freq_bins = h.n_fft // 2 + 1
    time_frames = 160 

    config_dict = {
        "input_info": [
            {"sample_size": [1, freq_bins, time_frames]}, 
            {"sample_size": [1, freq_bins, time_frames]}
        ],
        "compression": {
            "algorithm": "filter_pruning",
            "pruning_init": 0.0,
            "num_init_steps": num_init_steps,
            "pruning_flops_target": pruning_target,
            "pruning_steps": pruning_steps,
            "pruning_frequency": pruning_frequency,
            "filter_importance": filter_importance_type,
            
            "prune_first_conv": True,
            "prune_last_conv": True,
            "prune_downsample_convs": True,
            "prune_batch_norms": True,
            "all_weights": True
        }
    }

    nncf_config = NNCFConfig(config_dict)

    model.train()
    model.to(device)
    
    compression_ctrl, compressed_model = create_compressed_model(
        model, 
        nncf_config
    )

    optimizer = torch.optim.Adam(compressed_model.parameters(), lr=lr)
    compression_ctrl.scheduler.step()

    for epoch in range(epochs):
        total_loss = 0.0
        compression_ctrl.scheduler.epoch_step()
        
        for batch_idx, (clean_audio, noisy_audio) in enumerate(dataloader):
            clean_audio = clean_audio.to(device)
            noisy_audio = noisy_audio.to(device)

            clean_mag, clean_pha, clean_com = mag_pha_stft(
                clean_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )
            noisy_mag, noisy_pha, _ = mag_pha_stft(
                noisy_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )

            mag_g, pha_g, com_g = compressed_model(noisy_mag, noisy_pha)

            audio_g = mag_pha_istft(
                mag_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )
            mag_g_hat, _, com_g_hat = mag_pha_stft(
                audio_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )

            loss_mag = F.mse_loss(clean_mag, mag_g)
            loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, pha_g)
            loss_pha = loss_ip + loss_gd + loss_iaf
            loss_com = F.mse_loss(clean_com, com_g) * 2
            loss_stft = F.mse_loss(com_g, com_g_hat) * 2
            loss_time = F.l1_loss(clean_audio, audio_g)
            
            task_loss = (loss_mag * 0.9 + loss_pha * 0.3 + loss_com * 0.1 +
                         loss_stft * 0.1 + loss_time * 0.2)
            compression_loss = compression_ctrl.loss()
            loss = task_loss + compression_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            compression_ctrl.scheduler.step()
            
            total_loss += task_loss.item()

        avg_loss = total_loss / len(dataloader)
        stats = compression_ctrl.statistics()
        
        if hasattr(stats.filter_pruning, 'current_pruning_level'):
            current_sparsity = stats.filter_pruning.current_pruning_level * 100
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Sparsity: {current_sparsity:.2f}%")
        else:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            print(stats.filter_pruning.to_str())

    pruned_model = compression_ctrl.strip()
    pruned_model.eval()

    torch.save({'generator': pruned_model.state_dict()}, "nncf_pruned_model_new_new.pth")

    global_sparsity, total_params, zero_params = compute_global_sparsity(pruned_model)
    print(f"\nGlobal sparsity after pruning: {global_sparsity}")
    print(f"Total params: {total_params}, zero params: {zero_params}")
    print(f"Final pruned model size: {get_model_size_mb(pruned_model):.2f} MB")

    dummy_input = (torch.randn(1, 201, 160).to(device), torch.randn(1, 201, 160).to(device))
    torch.onnx.export(pruned_model, dummy_input, "nncf_pruned_model_new_new.onnx",
                  input_names=['amplitude', 'phase'],
                  output_names=['magnitude', 'phase_out'],
                  dynamic_axes={'amplitude': {2: 'time_frames'}, 'phase': {2: 'time_frames'},
                                'magnitude': {2: 'time_frames'}, 'phase_out': {2: 'time_frames'}})

    return pruned_model