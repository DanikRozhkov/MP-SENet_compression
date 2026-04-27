import os
import json
import argparse
from env import AttrDict
import torch
import soundfile as sf
import numpy as np
import inference
from cal_metrics.compute_metrics import compute_metrics

h = None
device = None

def compute_all_metrics(clean_dir, enhanced_dir, sr=16000):
    print("Start computing metrics")

    metrics_list = []
    print(clean_dir, enhanced_dir)
    for filename in os.listdir(enhanced_dir):
        if not filename.endswith('.wav'):
            continue
        enhanced_path = os.path.join(enhanced_dir, filename)
        clean_path = os.path.join(clean_dir, filename)
        if not os.path.exists(clean_path):
            print(f"Warning: clean file not found for {filename}")
            continue
        clean, sr_clean = sf.read(clean_path)
        enhanced, sr_enh = sf.read(enhanced_path)
        assert sr_clean == sr_enh, "Sampling rates differ"
        clean = clean[:len(clean)]
        enhanced = enhanced[:len(clean)]
        metrics = compute_metrics(clean, enhanced, sr_clean, 0)
        metrics_list.append(metrics)
    metrics_avg = np.mean(metrics_list, axis=0)
    return metrics_avg

def main():
    print("Initializing experiment process")

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_noisy_wavs_dir', default='VoiceBank+DEMAND/testset_noisy')
    parser.add_argument('--output_dir', default='VoiceBank+DEMAND/generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference.h = h
    inference.device = device
    inference.inference(a)

    clean_dir = "VoiceBank+DEMAND/wav_clean"   # путь к чистым файлам
    enhanced_dir = "VoiceBank+DEMAND/generated_files"   # та же папка, куда сохранил inference
    sr = h.sampling_rate   # из конфига

    metrics_avg = compute_all_metrics(clean_dir, enhanced_dir, sr)
    print(f"WB-PESQ: {metrics_avg[0]:.3f}, CSIG: {metrics_avg[1]:.3f}, CBAK: {metrics_avg[2]:.3f}, COVL: {metrics_avg[3]:.3f}, SSNR: {metrics_avg[4]:.3f}, STOI: {metrics_avg[5]:.3f}")


if __name__ == '__main__':
    main()