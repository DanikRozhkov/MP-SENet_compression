import os
import json
import argparse
from env import AttrDict
import torch
import soundfile as sf
import numpy as np
import inference
from compression.utils import compute_all_metrics, log_results


h = None
device = None

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

    metrics_avg = compute_all_metrics(clean_dir, enhanced_dir)
    print(f"WB-PESQ: {metrics_avg[0]:.3f}, CSIG: {metrics_avg[1]:.3f}, CBAK: {metrics_avg[2]:.3f}, COVL: {metrics_avg[3]:.3f}, SSNR: {metrics_avg[4]:.3f}, STOI: {metrics_avg[5]:.3f}")

    log_results(
        metrics_avg=metrics_avg,
        checkpoint_file=a.checkpoint_file,
        config_dict=json_config,
        device=device,
        experiment_id='baseline',
        method='none',
        compression_params=''
    )


if __name__ == '__main__':
    main()