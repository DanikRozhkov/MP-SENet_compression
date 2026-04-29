import os
import json
import yaml
import argparse
from env import AttrDict
import torch
import soundfile as sf
import numpy as np
import inference
from compression.utils import compute_all_metrics, log_results


h = None
device = None

def apply_compression(config):
    print(f"Applying {config['experiment']['method']} method")
    if config['experiment']['method'] == 'pruning':
        print(config['compression']['some_parameter'])
    elif config['experiment']['method'] == 'quantization':
        pass
    elif config['experiment']['method'] == 'baseline':
        pass
    else:
        print(f"Warning: no such compression method")

def main():
    print("Initializing experiment process")
    # Считываю аргументы из консоли - один конфиг
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    a = parser.parse_args()

    # Распаршиваю конфиг
    with open(a.config, 'r') as f:
        experiment_config = yaml.safe_load(f)

    config_file = os.path.join(os.path.split(experiment_config['data']['checkpoint_file'])[0], 'config.json')
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

    # Применяю сжатие
    apply_compression(config=experiment_config)

    # Запускаю инференс
    inference.h = h
    inference.device = device
    inference.inference(experiment_config['data']['input_noisy_wavs_dir'],
                        experiment_config['data']['output_dir'],
                        experiment_config['data']['checkpoint_file'])

    clean_dir = "VoiceBank+DEMAND/wav_clean"   # путь к чистым файлам
    enhanced_dir = "VoiceBank+DEMAND/generated_files"   # та же папка, куда сохранил inference

    # Считаю метрики
    metrics_avg = compute_all_metrics(clean_dir, enhanced_dir)
    print(f"WB-PESQ: {metrics_avg[0]:.3f}, CSIG: {metrics_avg[1]:.3f}, CBAK: {metrics_avg[2]:.3f}, COVL: {metrics_avg[3]:.3f}, SSNR: {metrics_avg[4]:.3f}, STOI: {metrics_avg[5]:.3f}")

    # Логирую результат
    log_results(
        metrics_avg=metrics_avg,
        checkpoint_file=experiment_config['data']['checkpoint_file'],
        config_dict=json_config,
        device=device,
        experiment_id=experiment_config['experiment']['id'],
        method=experiment_config['experiment']['method']
    )


if __name__ == '__main__':
    main()