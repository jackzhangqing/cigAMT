"""
Sample from a trained conditional Noise-DiT checkpoint.

Examples:
    python sample_noise.py --ckpt runs/noise_dit_v1/ckpts/latest.pt --n 100 --type_name Pulse --length 300
    python sample_noise.py --ckpt runs/noise_dit_v1/ckpts/latest.pt --n 200 --type_name all --lengths 100,300,750,1500,2048

--lengths 100, 150, 200, 300, 400, 450, 500, 600, 750, 1000, 1500, 2048

python sample_noise.py --ckpt runs/noise_dit_v1\ckpts\latest.pt 
--type_name all --lengths 100,300,750,1500,2048 
--n 100  # 生成样本总数，不是每个类型的数量，也不是每个长度的数量  是加在一起的数量   (一共10个噪声类型，12个长度类型)
--out_path runs/noise_dit_v1\generated\noise_samples.npz



python sample_noise.py --ckpt runs/noise_dit_v1\ckpts\latest.pt 
--type_name Pulse 
--length 300 
--n 50 
--out_path runs/noise_dit_v1\generated\pulse_len300.npz

NOISE_TYPE_NAMES = [
    "Pulse",
    "Quasi-Sinusoidal",
    "Square-wave",
    "Step-wave",
    "Triangular-wave",
    "Power-frequency",
    "White-noise",
    "Color-noise",
    "Harmonic",
    "Composite",
]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from diffusion import GaussianDiffusion
from noise_dataset import NOISE_TYPE_NAMES, TYPE_TO_ID
from noise_model import ConditionalNoiseDiT


def parse_lengths(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def load_model_from_ckpt(ckpt_path, use_ema=True, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    config = ckpt["config"]
    model = ConditionalNoiseDiT(
        in_channels=config["in_channels"],
        input_length=config["input_length"],
        patch_size=config["patch_size"],
        hidden_dim=config["hidden_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        num_noise_types=config["num_noise_types"],
        condition_on_length=config.get("condition_on_length", True),
        condition_on_source=config.get("condition_on_source", False),
    ).to(device)

    if use_ema and "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
        print("loaded EMA weights")
    else:
        model.load_state_dict(ckpt["model"])
        print("loaded model weights")
    model.eval()

    diffusion = GaussianDiffusion(
        num_timesteps=config["num_timesteps"],
        schedule=config["beta_schedule"],
        device=device,
    )
    return model, diffusion, config


def build_conditions(args, config, batch_size, start_idx, device):
    if args.type_name == "all":
        type_ids = [(start_idx + i) % config["num_noise_types"] for i in range(batch_size)]
    else:
        if args.type_name not in TYPE_TO_ID:
            raise ValueError(f"Unknown type_name: {args.type_name}. Choices: {NOISE_TYPE_NAMES}")
        type_ids = [TYPE_TO_ID[args.type_name]] * batch_size

    lengths = args.lengths or [args.length]
    # 遍历当前长度的全部类别后再切换长度，使用全局索引保持跨 batch 连续。
    types_per_length = config["num_noise_types"] if args.type_name == "all" else 1
    length_values = [
        lengths[((start_idx + i) // types_per_length) % len(lengths)]
        for i in range(batch_size)
    ]

    type_id = torch.tensor(type_ids, dtype=torch.long, device=device)
    length = torch.tensor(length_values, dtype=torch.long, device=device)
    length_ratio = length.float() / float(config["target_length"])
    source_id = torch.full((batch_size,), args.source_id, dtype=torch.long, device=device)
    return type_id, length, length_ratio, source_id


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Sample conditional AMT noise")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--type_name", default="all", help="'all' or one of the noise type names")
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--lengths", type=parse_lengths, default=None)
    parser.add_argument("--source_id", type=int, default=1, help="0=real condition, 1=synthetic condition")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--out_path", default="noise_samples.npz")
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--no_ema", dest="use_ema", action="store_false")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    model, diffusion, config = load_model_from_ckpt(args.ckpt, args.use_ema, device)
    print(f"sampling n={args.n}, steps={args.num_steps}, shape=(1,{config['input_length']})")

    all_samples = []
    all_type_id = []
    all_length = []
    all_source_id = []
    n_batches = (args.n + args.batch_size - 1) // args.batch_size

    for bi in range(n_batches):
        start_idx = bi * args.batch_size
        bs = min(args.batch_size, args.n - start_idx)
        type_id, length, length_ratio, source_id = build_conditions(args, config, bs, start_idx, device)
        shape = (bs, config["in_channels"], config["input_length"])

        if args.num_steps >= config["num_timesteps"]:
            samples = diffusion.sample(  # 直接采样,不压缩时间步  也可以ddim_sample：采样步数少 → 更快
                model,
                shape,
                device=device,
                progress=True,
                type_id=type_id,
                length_ratio=length_ratio,
                source_id=source_id,
            )
        else:
            samples = diffusion.ddim_sample(
                model,
                shape,
                num_steps=args.num_steps,
                eta=args.eta,
                device=device,
                progress=True,
                type_id=type_id,
                length_ratio=length_ratio,
                source_id=source_id,
            )

        all_samples.append(samples.cpu().numpy())
        all_type_id.append(type_id.cpu().numpy())
        all_length.append(length.cpu().numpy())
        all_source_id.append(source_id.cpu().numpy())

    samples_np = np.concatenate(all_samples, axis=0)
    type_np = np.concatenate(all_type_id, axis=0)
    length_np = np.concatenate(all_length, axis=0)
    source_np = np.concatenate(all_source_id, axis=0)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".h5":
        with h5py.File(out_path, "w") as f:
            f.create_dataset("samples", data=samples_np.astype(np.float32), compression="gzip", compression_opts=4)
            f.create_dataset("type_id", data=type_np.astype(np.int64))
            f.create_dataset("length", data=length_np.astype(np.int64))
            f.create_dataset("source_id", data=source_np.astype(np.int64))
            f.attrs["type_names"] = json.dumps(NOISE_TYPE_NAMES)
            f.attrs["ckpt"] = str(args.ckpt)
            f.attrs["config"] = json.dumps(config)
    else:
        np.savez_compressed(
            out_path,
            samples=samples_np,
            type_id=type_np,
            length=length_np,
            source_id=source_np,
            type_names=np.asarray(NOISE_TYPE_NAMES, dtype=str),
            config=np.asarray(json.dumps(config), dtype=str),
        )

    print(f"saved: {out_path}")
    print(f"file size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        png_path = out_path.with_suffix(".png")
        rows = min(args.n, 20)
        fig, axes = plt.subplots(rows, 1, figsize=(14, max(2.0, rows * 1.35)), sharex=True)
        if rows == 1:
            axes = [axes]
        for i in range(rows):
            axes[i].plot(samples_np[i, 0], color="#1f2937", linewidth=0.7)
            axes[i].grid(alpha=0.25)
            axes[i].set_ylabel(f"{NOISE_TYPE_NAMES[int(type_np[i])][:10]}\nL={int(length_np[i])}", fontsize=8)
        axes[-1].set_xlabel("sample index")
        fig.tight_layout()
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        print(f"preview: {png_path}")
    except Exception as exc:
        print(f"plot skipped: {exc}")


if __name__ == "__main__":
    main()
