import os
import shutil
import argparse
import soundfile as sf
import numpy as np
import torch
import csv
from datetime import datetime
from models.model import MPNet
from env import AttrDict
from calflops import calculate_flops
from cal_metrics.compute_metrics import compute_metrics

def generate_test_dataset(noisy_dir="VoiceBank+DEMAND/wav_noisy",
                          list_file="VoiceBank+DEMAND/test.txt",
                          target_dir="VoiceBank+DEMAND/testset_noisy"):
    """
    Копирует шумные .wav файлы из source_noisy_dir в target_noisy_dir
    Имена файлов берутся из list_file (первое поле до '|', добавляется .wav)
    """
    os.makedirs(target_dir, exist_ok=True)
    
    with open(list_file, 'r') as f:
        lines = f.readlines()
    
    copied_count = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        file_id = line.split('|')[0].strip()
        filename = file_id + '.wav'
        
        src = os.path.join(noisy_dir, filename)
        dst = os.path.join(target_dir, filename)
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied_count += 1
        else:
            print(f"Warning: {src} not found")
    
    print(f"Copied {copied_count} noisy files to {target_dir}")


def compute_all_metrics(clean_dir, enhanced_dir):
    """
    Подсчитывает метрики PESQ, CSIG, CBAK, COVL, SSNR, STOI
    clean_dir - путь к папке с эталонными данными
    enhanced_dir - путь к папке с результатми работы модели
    """
    print("Start computing metrics")

    metrics_list = []
    for filename in os.listdir(enhanced_dir):
        if not filename.endswith('.wav'):
            continue
        enhanced_path = os.path.join(enhanced_dir, filename)
        clean_path = os.path.join(clean_dir, filename)
        if not os.path.exists(clean_path):
            print(f"Warning: clean file not found for {filename}")
            continue
        clean, sr_clean = sf.read(clean_path)
        enhanced, _ = sf.read(enhanced_path)
        clean = clean[:len(clean)]
        enhanced = enhanced[:len(clean)]
        metrics = compute_metrics(clean, enhanced, sr_clean, 0)
        metrics_list.append(metrics)

    metrics_avg = np.mean(metrics_list, axis=0)
    return metrics_avg


def count_params_flops_macs(model, input_shape_amp, input_shape_pha, device='cpu'):
    """
    Подсчитывает метрики Params, FLOPs и MACs
    """
    model.eval()
    amp = torch.randn(input_shape_amp).to(device)
    pha = torch.randn(input_shape_pha).to(device)

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, amp, pha):
            out = self.model(amp, pha)
            return out[0]

    wrapped_model = Wrapper(model)

    flops, macs, params = calculate_flops(
        model=wrapped_model,
        args=[amp, pha],
        print_results=False
    )
    return params, flops, macs

def log_results(model, metrics_avg, checkpoint_file, device='cpu', 
                experiment_id='baseline', method='none', file_path="experiments/results.csv"):
    """
    Записывает результаты эксперимента в файл experiments/results.csv
    """
    input_shape_amp = (1, 201, 160) # TODO: не хардкодить это
    input_shape_pha = (1, 201, 160) # TODO: не хардкодить это
    params, flops, macs = count_params_flops_macs(model, input_shape_amp, input_shape_pha, device)
    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'experiment_id': experiment_id,
        'method': method,
        'checkpoint': checkpoint_file,
        'params': params,
        'flops': flops,
        'macs': macs,
        'PESQ': f"{metrics_avg[0]:.5f}",
        'CSIG': f"{metrics_avg[1]:5f}",
        'CBAK': f"{metrics_avg[2]:.5f}",
        'COVL': f"{metrics_avg[3]:.5f}",
        'SSNR': f"{metrics_avg[4]:.5f}",
        'STOI': f"{metrics_avg[5]:.5f}",
    }

    csv_path = file_path
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Results logged to {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Utilities for MP-SENet experiments")
    subparsers = parser.add_subparsers(dest='command', required=True)
    parser_noisy = subparsers.add_parser('gendataset', help='Generate noisy test dataset')
    parser_noisy.add_argument('--source_noisy_dir', default='VoiceBank+DEMAND/wav_noisy',
                              help='Source directory with all noisy wavs')
    parser_noisy.add_argument('--list_file', default='VoiceBank+DEMAND/test.txt',
                              help='Text file listing test files (first field)')
    parser_noisy.add_argument('--target_noisy_dir', default='VoiceBank+DEMAND/testset_noisy',
                              help='Output directory for noisy test subset')

    args = parser.parse_args()

    if args.command == 'gendataset':
        generate_test_dataset(
            noisy_dir=args.source_noisy_dir,
            list_file=args.list_file,
            target_dir=args.target_noisy_dir
        )
    else:
        parser.print_help()