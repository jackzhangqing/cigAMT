"""Run the original single-process or DDP trainer using a portable saved config."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--synthetic-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--ddp', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config.update(data_root=str(args.data_root), synthetic_root=str(args.synthetic_root),
                  out_dir=str(args.output), resume_from=None)
    if args.ddp:
        from train_noise_ddp import train
    else:
        from train_noise import train
    train(config)


if __name__ == '__main__': main()
