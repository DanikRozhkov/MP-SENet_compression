from __future__ import absolute_import, division, print_function, unicode_literals
import sys
sys.path.append("..")
import glob
import os
import argparse
import json
from re import S
import torch
import librosa
from env import AttrDict
from dataset import mag_pha_stft, mag_pha_istft
from models.model import MPNet
import soundfile as sf
from rich.progress import track

h = None
device = None
use_amp = False

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    # Вылезает предупреждение, что скоро weights_only по умолчанию будет True, поэтому укажем значение явно
    checkpoint_dict = torch.load(filepath, map_location=device, weights_only=False)
    print("Complete.")
    return checkpoint_dict

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

def inference(input_noisy_wavs_dir, output_dir, checkpoint_file, model=None):
    # Если передана модель то параметр checkpoint_file не используется
    if model is None:
        model = MPNet(h).to(device)
        state_dict = load_checkpoint(checkpoint_file, device)
        model.load_state_dict(state_dict['generator'])
    else:
        model = model.to(device)

    test_indexes = os.listdir(input_noisy_wavs_dir)

    os.makedirs(output_dir, exist_ok=True)

    model.eval()

    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    amp_dtype = torch.float16 if device_type == 'cuda' else torch.bfloat16
    is_amp_enabled = getattr(model, 'use_amp', False)
    if is_amp_enabled:
        print(f"Running inference with AMP enabled ({amp_dtype}).")

    with torch.no_grad():
        for index in track(test_indexes):
            noisy_wav, _ = librosa.load(os.path.join(input_noisy_wavs_dir, index), sr=h.sampling_rate)
            noisy_wav = torch.FloatTensor(noisy_wav).to(device)
            norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to(device)
            noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
            noisy_amp, noisy_pha, noisy_com = mag_pha_stft(noisy_wav, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            
            with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=is_amp_enabled):
                amp_g, pha_g, com_g = model(noisy_amp, noisy_pha)
            amp_g = amp_g.float()
            pha_g = pha_g.float()
            
            audio_g = mag_pha_istft(amp_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            audio_g = audio_g / norm_factor

            output_file = os.path.join(output_dir, index)

            sf.write(output_file, audio_g.squeeze().cpu().numpy(), h.sampling_rate, 'PCM_16')


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_noisy_wavs_dir', default='VoiceBank+DEMAND/testset_noisy')
    parser.add_argument('--output_dir', default='../generated_files')
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

    inference(a.input_noisy_wavs_dir, a.output_dir, a.checkpoint_file)


if __name__ == '__main__':
    main()
