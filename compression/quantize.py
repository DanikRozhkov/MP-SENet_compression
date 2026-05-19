import copy
import torch
import random
from torch.ao.quantization import QConfig, QConfigMapping
from torch.ao.quantization.observer import HistogramObserver, MinMaxObserver
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.utils.data import DataLoader
import torch.ao.quantization as quant
from compression.utils import get_model_size_mb
from dataset import Dataset, get_dataset_filelist, mag_pha_stft

def quantization(model, config, device='cpu'):
    torch.backends.quantized.engine = 'fbgemm'
    print(f"Quantization engine set to: {torch.backends.quantized.engine}")
    print(f"Base model size: {get_model_size_mb(model):.2f} MB")
    quant_mode = config.get('mode', 'dynamic')
    quant_dtype = getattr(torch, config.get('dtype', 'qint8'))
    model.eval()

    if quant_mode == 'dynamic':
        quantized_model = quant.quantize_dynamic(
            model,
            qconfig_spec={torch.nn.Linear},
            dtype=quant_dtype,
            inplace=False
        )
        print("Dynamic quantization completed")
        print(f"Quantized model size: {get_model_size_mb(quantized_model):.2f} MB")
        return quantized_model

    elif quant_mode == 'static':
        class Args:
            input_training_file = "VoiceBank+DEMAND/training.txt"
            input_validation_file = "VoiceBank+DEMAND/test.txt"
            input_clean_wavs_dir = "VoiceBank+DEMAND/wav_clean"
            input_noisy_wavs_dir = "VoiceBank+DEMAND/wav_noisy"
        args = Args()
        all_train_indexes, _ = get_dataset_filelist(args)
        train_indexes = random.sample(all_train_indexes, 200)
        dataset = Dataset(
            training_indexes=train_indexes,
            clean_wavs_dir=args.input_clean_wavs_dir,
            noisy_wavs_dir=args.input_noisy_wavs_dir,
            segment_size=32000,
            sampling_rate=16000,
            split=True,
            shuffle=True,
            n_cache_reuse=0,
            device=device
        )
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

        model_to_quantize = copy.deepcopy(model)
        model_to_quantize.eval()
        model_to_quantize.to(device)

        my_qconfig = QConfig(
            activation=HistogramObserver.with_args(reduce_range=True),
            weight=MinMaxObserver.with_args(
                dtype=torch.qint8, 
                qscheme=torch.per_tensor_symmetric
            )
        )
        custom_qconfig_mapping = QConfigMapping().set_global(my_qconfig).set_object_type(torch.nn.GRU, None)

        example_amp = torch.randn(1, 201, 160).to(device)
        example_pha = torch.randn(1, 201, 160).to(device)
        example_inputs = (example_amp, example_pha)

        model_prepared = prepare_fx(model_to_quantize, custom_qconfig_mapping, example_inputs=example_inputs)
        model_prepared.to(device)

        with torch.no_grad():
            for batch in dataloader:
                _, noisy_audio = batch
                if noisy_audio.dim() == 1:
                    noisy_audio = noisy_audio.unsqueeze(0)
                noisy_amp, noisy_pha, _ = mag_pha_stft(noisy_audio, 400, 100, 400, 0.3)
                noisy_amp = noisy_amp.to(device)
                noisy_pha = noisy_pha.to(device)
                model_prepared(noisy_amp, noisy_pha)

        quantized_model = convert_fx(model_prepared)

        def apply_dynamic_quantize_to_gru(module):
            for name, child in module.named_children():
                if isinstance(child, torch.nn.GRU):
                    new_gru = quant.quantize_dynamic(child, {torch.nn.GRU}, dtype=torch.qint8, inplace=False)
                    setattr(module, name, new_gru)
                    print(f"Dynamic quantization of GRU: {name}")
                else:
                    apply_dynamic_quantize_to_gru(child)
        apply_dynamic_quantize_to_gru(quantized_model)

        print(f"Model size: {get_model_size_mb(quantized_model):.2f} MB")
        return quantized_model

    else:
        raise ValueError(f"Unknown method: {quant_mode}")