import os
import shutil
import argparse


def generate_test_dataset(noisy_dir="VoiceBank+DEMAND/wav_noisy",
                          list_file="VoiceBank+DEMAND/test.txt",
                          target_dir="VoiceBank+DEMAND/testset_noisy"):
    """
    Копирует шумные .wav файлы из source_noisy_dir в target_noisy_dir.
    Имена файлов берутся из list_file (первое поле до '|', добавляется .wav).
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