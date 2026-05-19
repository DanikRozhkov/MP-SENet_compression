import os
import json
import yaml
import argparse
from env import AttrDict
import torch
import inference
from models.model import MPNet
from compression.utils import compute_all_metrics, log_results, get_model_size_mb
from compression.prune import unstructured_pruning
from compression.quantize import quantization
from compression.finetune import fine_tune_model


h = None
device = None

def apply_compression(config, model, device='cpu'):
    method = config['experiment']['method']
    compression_parameters = config.get('compression', {})
    if method == 'pruning':
        pruning_type = config['compression'].get('type', 'unstructured')

        if pruning_type == 'unstructured':
            model = unstructured_pruning(model, compression_parameters)
        else:
            # model = structured_pruning(model, compression_parameters)
            pass
            
        print(f"Applying {pruning_type} {method} method with parameters: {compression_parameters}")
        return model
    
    elif method == 'quantization':
        model = quantization(model, compression_parameters, device=device)
        return model
    
    elif method == 'baseline':
        return model
    
    else:
        print(f"Warning: no such compression method")
        return model
        

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

    # Загружаю модель
    model = MPNet(h).to(device)
    checkpoint = torch.load(experiment_config['data']['checkpoint_file'], map_location='cpu')
    model.load_state_dict(checkpoint['generator'])

    # Применяю сжатие
    model = apply_compression(config=experiment_config, model=model, device=device)

    if experiment_config['compression']['finetune']:
        model = fine_tune_model(
            model=model,
            device=device,
            checkpoint_path=None,
            h=h,
            num_files=experiment_config['compression']['num_files'],
            batch_size=experiment_config['compression']['batch_size'],
            epochs=experiment_config['compression']['epochs'],
            lr=experiment_config['compression']['lr'],
            seed=h.seed,
            qat=True # TODO: Потом поменять
        )

    # Запускаю инференс
    inference.h = h
    inference.device = device
    inference.inference(experiment_config['data']['input_noisy_wavs_dir'],
                        experiment_config['data']['output_dir'],
                        experiment_config['data']['checkpoint_file'], # TODO: как будто этот параметр можно просто убрать, перепроверить
                        model=model)

    clean_dir = "VoiceBank+DEMAND/wav_clean"   # путь к чистым файлам
    enhanced_dir = experiment_config['data']['output_dir']   # та же папка, куда сохранил inference

    # Считаю метрики
    metrics_avg = compute_all_metrics(clean_dir, enhanced_dir)
    print(f"WB-PESQ: {metrics_avg[0]:.3f}, CSIG: {metrics_avg[1]:.3f}, CBAK: {metrics_avg[2]:.3f}, COVL: {metrics_avg[3]:.3f}, SSNR: {metrics_avg[4]:.3f}, STOI: {metrics_avg[5]:.3f}")

    # Логирую результат
    log_results(
        model=model,
        metrics_avg=metrics_avg,
        checkpoint_file=experiment_config['data']['checkpoint_file'],
        device=device,
        experiment_id=experiment_config['experiment']['id'],
        method=experiment_config['experiment']['method'],
        file_path=experiment_config['data']['results_file']
    )


if __name__ == '__main__':
    main()