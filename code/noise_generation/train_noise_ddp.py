"""
Train conditional Noise-DiT with optional DistributedDataParallel (DDP).

Single GPU:
    python train_noise_ddp.py --data_root /data/0-noise --synthetic_root /data/0-noise/Synthetic-Norm

Multi-GPU:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train_noise_ddp.py \
      --data_root /data/0-noise \
      --synthetic_root /data/0-noise/Synthetic-Norm \
      --out_dir runs/noise_dit_ddp \
      --batch_size 64


CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 --master_port=29512 train_noise_ddp.py \
  --data_root /your/path/0-noise \
  --synthetic_root /your/path/0-noise/Synthetic-Norm \
  --out_dir runs/noise_dit_ddp_gpu23 \
  --batch_size 64 \
  --num_workers 2 \
  --num_steps 200000


CUDA_VISIBLE_DEVICES=2  python train_noise_ddp.py \
  --data_root /your/path/0-noise \
  --synthetic_root /your/path/0-noise/Synthetic-Norm \
  --out_dir runs/noise_dit_single \
  --batch_size 128


CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 --master_port=29512 train_noise_ddp.py \
  --data_root /your/path/0-noise \
  --synthetic_root /your/path/0-noise/Synthetic-Norm \
  --out_dir runs/noise_dit_ddp_gpu23 \
  --batch_size 64 \
  --resume_from runs/noise_dit_ddp_gpu23/ckpts/latest.pt

Note: batch_size is per GPU. With 2 GPUs and batch_size=64, the global batch size is 128.
By default, train on both real and synthetic clips at every available length <= 2048.
Short clips are randomly embedded in 2048 points and conditioned on length_ratio.
Use --train_lengths all (the default) or a comma-separated subset such as 100,300,2048.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler

from diffusion import GaussianDiffusion
from ema import EMA
from noise_dataset import NOISE_TYPE_NAMES, NoiseNPYDataset, SOURCE_TO_ID
from noise_model import ConditionalNoiseDiT


DEFAULT_CONFIG = {
    # Data
    "data_root": r"noise_data",
    "synthetic_root": r"noise_data/Synthetic-Norm",
    "include_real": True,
    "include_synthetic": True,
    "target_length": 2048,
    "train_lengths": None,  # All available original lengths up to target_length.
    "data_normalize": "rms",
    "random_embed": True,
    "flip_prob": 0.5,
    "sign_flip_prob": 0.5,
    "balance_type": True,
    "balance_source": True,
    "real_weight": 0.5,  # 降低实测样本相对于合成样本的采样权重。
    "synthetic_weight": 1.0,

    # Model
    "in_channels": 1,
    "input_length": 2048,
    "patch_size": 8,
    "hidden_dim": 384,
    "depth": 12,
    "num_heads": 6,
    "mlp_ratio": 4.0,
    "num_noise_types": 10,
    "condition_on_length": True,
    "condition_on_source": False,

    # Diffusion
    "num_timesteps": 1000,
    "beta_schedule": "linear",

    # Optimizer/training
    "batch_size": 128,      # per GPU under DDP
    "lr": 1e-4,
    "weight_decay": 0.0,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "grad_clip": 1.0,
    "num_steps": 200_000,
    "warmup_steps": 1000,
    "ema_decay": 0.9999,
    "mixed_precision": "bf16",
    "num_workers": 4,      # per process

    # Logging/saving
    "log_interval": 50,
    "sample_interval": 5000,
    "ckpt_interval": 10_000,
    "n_samples_to_generate": 20,
    "sample_lengths": [100, 300, 750, 1500, 2048],
    "sample_ddim_steps": 50,

    # Output
    "out_dir": "runs/noise_dit_ddp_all_lengths",
    "resume_from": None,
    "seed": 42,
}


def setup_ddp():
    """Return rank, world_size, local_rank, device, and whether DDP is enabled."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"
        dist.init_process_group(backend=backend)
        return rank, world_size, local_rank, device, True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_rank = 0
    return 0, 1, local_rank, device, False


def cleanup_ddp(is_ddp: bool):
    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def barrier(is_ddp: bool):
    if is_ddp and dist.is_initialized():
        dist.barrier()


def reduce_mean(value: torch.Tensor, world_size: int, is_ddp: bool) -> torch.Tensor:
    if not is_ddp:
        return value.detach()
    value = value.detach().clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value /= world_size
    return value


def get_lr(step, warmup_steps, total_steps, base_lr, min_lr_ratio=0.1):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    min_lr = base_lr * min_lr_ratio
    return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def make_rank_weighted_sampler(dataset, config, rank, world_size):
    """
    WeightedRandomSampler with class/source balancing.
    Under DDP, each rank draws len(dataset)/world_size samples with a different seed.
    """
    type_counts = Counter(item.type_id for item in dataset.items)
    source_counts = Counter(item.source_id for item in dataset.items)
    source_scale = {
        SOURCE_TO_ID["real"]: float(config["real_weight"]),
        SOURCE_TO_ID["synthetic"]: float(config["synthetic_weight"]),
    }

    weights = []
    for item in dataset.items:
        w = 1.0
        if config["balance_type"]:
            w *= 1.0 / max(type_counts[item.type_id], 1)
        if config["balance_source"]:
            w *= 1.0 / max(source_counts[item.source_id], 1)
        w *= source_scale.get(item.source_id, 1.0)
        weights.append(w)

    if world_size > 1:
        num_samples = int(math.ceil(len(dataset) / world_size))
    else:
        num_samples = len(dataset)

    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]) + 1009 * rank)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )


def build_dataloader(config, rank=0, world_size=1):
    dataset = NoiseNPYDataset(
        data_root=config["data_root"],
        synthetic_root=config["synthetic_root"],
        target_length=config["target_length"],
        include_real=config["include_real"],
        include_synthetic=config["include_synthetic"],
        max_real_original_len=config["target_length"],
        selected_lengths=config["train_lengths"],
        normalize=config["data_normalize"],
        random_embed=config["random_embed"],
        flip_prob=config["flip_prob"],
        sign_flip_prob=config["sign_flip_prob"],
    )
    sampler = make_rank_weighted_sampler(dataset, config, rank, world_size)
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        sampler=sampler,
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=config["num_workers"] > 0,
    )
    return dataset, loader


def build_model(config, device):
    return ConditionalNoiseDiT(
        in_channels=config["in_channels"],
        input_length=config["input_length"],
        patch_size=config["patch_size"],
        hidden_dim=config["hidden_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        num_noise_types=config["num_noise_types"],
        condition_on_length=config["condition_on_length"],
        condition_on_source=config["condition_on_source"],
    ).to(device)


def save_checkpoint(path, step, model, ema, optimizer, scaler, config):
    raw_model = unwrap_model(model)
    ckpt = {
        "step": step,
        "model": raw_model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "noise_type_names": NOISE_TYPE_NAMES,
    }
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    torch.save(ckpt, path)
    print(f"  saved checkpoint: {path}")


def load_checkpoint(path, model, ema, optimizer, scaler, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    raw_model = unwrap_model(model)
    raw_model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt["step"])


@torch.no_grad()
def sample_and_save(ema, diffusion, config, step, out_dir, device):
    ema.shadow_model.eval()

    type_ids = []
    length_values = []
    sample_lengths = list(config["sample_lengths"])
    n = int(config["n_samples_to_generate"])
    for i in range(n):
        type_ids.append(i % config["num_noise_types"])
        length_values.append(sample_lengths[(i + i // config["num_noise_types"]) % len(sample_lengths)])

    type_id = torch.tensor(type_ids, dtype=torch.long, device=device)
    length = torch.tensor(length_values, dtype=torch.long, device=device)
    length_ratio = length.float() / float(config["target_length"])
    source_id = torch.ones_like(type_id)

    shape = (n, config["in_channels"], config["input_length"])
    samples = diffusion.ddim_sample(
        ema.shadow_model,
        shape,
        num_steps=config["sample_ddim_steps"],
        eta=0.0,
        device=device,
        progress=False,
        type_id=type_id,
        length_ratio=length_ratio,
        source_id=source_id,
    )

    samples_np = samples.cpu().numpy()
    type_np = type_id.cpu().numpy()
    length_np = length.cpu().numpy()

    save_dir = Path(out_dir) / "samples"
    save_dir.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(
        save_dir / f"samples_step{step:07d}.npz",
        samples=samples_np,
        type_id=type_np,
        length=length_np,
        type_names=np.asarray(NOISE_TYPE_NAMES, dtype=object),
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = min(n, 20)
        fig, axes = plt.subplots(rows, 1, figsize=(14, max(2.0, rows * 1.35)), sharex=True)
        if rows == 1:
            axes = [axes]
        for i in range(rows):
            axes[i].plot(samples_np[i, 0], color="#1f2937", linewidth=0.7)
            axes[i].grid(alpha=0.25)
            axes[i].set_ylabel(f"{NOISE_TYPE_NAMES[type_np[i]][:10]}\nL={length_np[i]}", fontsize=8)
        axes[-1].set_xlabel("sample index")
        fig.suptitle(f"Noise-DiT EMA samples @ step {step}")
        fig.tight_layout()
        fig.savefig(save_dir / f"samples_step{step:07d}.png", dpi=120)
        plt.close(fig)
    except Exception as exc:
        print(f"  sample plot skipped: {exc}")


def train(config):
    rank, world_size, local_rank, device, is_ddp = setup_ddp()

    torch.manual_seed(int(config["seed"]) + rank)
    np.random.seed(int(config["seed"]) + rank)

    out_dir = Path(config["out_dir"])
    if is_main_process(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "ckpts").mkdir(exist_ok=True)
        (out_dir / "samples").mkdir(exist_ok=True)
        with (out_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    barrier(is_ddp)

    dataset, loader = build_dataloader(config, rank=rank, world_size=world_size)
    info = dataset.get_info()
    if is_main_process(rank):
        with (out_dir / "dataset_info.json").open("w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        print("=" * 70)
        print("Conditional Noise-DiT DDP training")
        print("=" * 70)
        print(f"world_size: {world_size}")
        print(f"device: {device}")
        print(f"per-GPU batch_size: {config['batch_size']}")
        print(f"global batch_size: {config['batch_size'] * world_size}")
        print(f"samples: {len(dataset):,}")
        print(f"type distribution: {info['type_distribution']}")
        print(f"source distribution: {info['source_distribution']}")
        print(f"length distribution: {info['length_distribution']}")

    model = build_model(config, device)
    if is_main_process(rank):
        print(f"parameters: {count_params(model) / 1e6:.2f} M")

    if is_ddp:
        ddp_kwargs = {}
        if device.type == "cuda":
            ddp_kwargs.update(device_ids=[local_rank], output_device=local_rank)
        model = DDP(model, **ddp_kwargs)

    raw_model = unwrap_model(model)
    ema = EMA(raw_model, decay=config["ema_decay"])
    diffusion = GaussianDiffusion(
        num_timesteps=config["num_timesteps"],
        schedule=config["beta_schedule"],
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        betas=(config["adam_beta1"], config["adam_beta2"]),
        weight_decay=config["weight_decay"],
    )

    use_amp = config["mixed_precision"] != "none" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if config["mixed_precision"] == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler() if use_amp and config["mixed_precision"] == "fp16" else None

    start_step = 0
    if config["resume_from"]:
        if is_main_process(rank):
            print(f"resume from: {config['resume_from']}")
        start_step = load_checkpoint(config["resume_from"], model, ema, optimizer, scaler, device)
        if is_main_process(rank):
            print(f"resumed step: {start_step}")

    loss_log_path = out_dir / "loss_log.csv"
    if is_main_process(rank) and (start_step == 0 or not loss_log_path.exists()):
        with loss_log_path.open("w", encoding="utf-8") as f:
            f.write("step,loss,avg_loss,lr,steps_per_sec\n")

    model.train()
    step = start_step
    data_iter = iter(loader)
    loss_window = []
    last_log_time = time.time()

    while step < config["num_steps"]:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        x_0 = batch["noise"].to(device, non_blocking=True)
        type_id = batch["type_id"].to(device, non_blocking=True)
        length_ratio = batch["length_ratio"].to(device, non_blocking=True)
        source_id = batch["source_id"].to(device, non_blocking=True)

        cur_lr = get_lr(step, config["warmup_steps"], config["num_steps"], config["lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        optimizer.zero_grad(set_to_none=True)
        model_kwargs = {
            "type_id": type_id,
            "length_ratio": length_ratio,
            "source_id": source_id,
        }

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                loss = diffusion.training_loss(model, x_0, model_kwargs=model_kwargs)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                optimizer.step()
        else:
            loss = diffusion.training_loss(model, x_0, model_kwargs=model_kwargs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()

        ema.update(raw_model)
        step += 1

        loss_avg_rank = reduce_mean(loss, world_size, is_ddp)
        loss_window.append(float(loss_avg_rank.item()))

        if step % config["log_interval"] == 0 and is_main_process(rank):
            avg_loss = sum(loss_window[-100:]) / min(len(loss_window), 100)
            elapsed = time.time() - last_log_time
            steps_per_sec = config["log_interval"] / max(elapsed, 1e-6)
            last_log_time = time.time()
            print(
                f"step {step:>7,} | loss {loss_avg_rank.item():.4f} | "
                f"avg {avg_loss:.4f} | lr {cur_lr:.2e} | {steps_per_sec:.2f} step/s"
            )
            with loss_log_path.open("a", encoding="utf-8") as f:
                f.write(f"{step},{loss_avg_rank.item():.6f},{avg_loss:.6f},{cur_lr:.8e},{steps_per_sec:.4f}\n")

        if step % config["sample_interval"] == 0:
            if is_main_process(rank):
                print(f"sampling @ step {step} ...")
                sample_and_save(ema, diffusion, config, step, out_dir, device)
            barrier(is_ddp)
            model.train()

        if step % config["ckpt_interval"] == 0:
            if is_main_process(rank):
                ckpt_path = out_dir / "ckpts" / f"ckpt_step{step:07d}.pt"
                save_checkpoint(ckpt_path, step, model, ema, optimizer, scaler, config)
                latest_path = out_dir / "ckpts" / "latest.pt"
                save_checkpoint(latest_path, step, model, ema, optimizer, scaler, config)
            barrier(is_ddp)

    if is_main_process(rank):
        save_checkpoint(out_dir / "ckpts" / "final.pt", step, model, ema, optimizer, scaler, config)
        print("training complete")
    barrier(is_ddp)
    cleanup_ddp(is_ddp)


def parse_int_list(text: str):
    if text is None or str(text).strip().lower() == "all":
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Train conditional Noise-DiT with optional DDP")

    parser.add_argument("--data_root", type=str, default=DEFAULT_CONFIG["data_root"])
    parser.add_argument("--synthetic_root", type=str, default=DEFAULT_CONFIG["synthetic_root"])
    parser.add_argument("--include_real", action="store_true", default=DEFAULT_CONFIG["include_real"])
    parser.add_argument("--no_real", dest="include_real", action="store_false")
    parser.add_argument("--include_synthetic", action="store_true", default=DEFAULT_CONFIG["include_synthetic"])
    parser.add_argument("--no_synthetic", dest="include_synthetic", action="store_false")
    parser.add_argument(
        "--train_lengths", type=parse_int_list, default=DEFAULT_CONFIG["train_lengths"],
        help="Original lengths to train on: 'all' (default) or a comma-separated list, e.g. 100,300,2048.",
    )
    parser.add_argument("--data_normalize", choices=["rms", "maxabs", "none"], default=DEFAULT_CONFIG["data_normalize"])
    parser.add_argument("--real_weight", type=float, default=DEFAULT_CONFIG["real_weight"])
    parser.add_argument("--synthetic_weight", type=float, default=DEFAULT_CONFIG["synthetic_weight"])
    parser.add_argument("--condition_on_length", action="store_true", default=DEFAULT_CONFIG["condition_on_length"])
    parser.add_argument("--no_length_condition", dest="condition_on_length", action="store_false")
    parser.add_argument("--condition_on_source", action="store_true", default=DEFAULT_CONFIG["condition_on_source"])

    parser.add_argument("--hidden_dim", type=int, default=DEFAULT_CONFIG["hidden_dim"])
    parser.add_argument("--depth", type=int, default=DEFAULT_CONFIG["depth"])
    parser.add_argument("--num_heads", type=int, default=DEFAULT_CONFIG["num_heads"])
    parser.add_argument("--patch_size", type=int, default=DEFAULT_CONFIG["patch_size"])

    parser.add_argument("--num_timesteps", type=int, default=DEFAULT_CONFIG["num_timesteps"])
    parser.add_argument("--beta_schedule", choices=["linear", "cosine"], default=DEFAULT_CONFIG["beta_schedule"])

    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--num_steps", type=int, default=DEFAULT_CONFIG["num_steps"])
    parser.add_argument("--warmup_steps", type=int, default=DEFAULT_CONFIG["warmup_steps"])
    parser.add_argument("--ema_decay", type=float, default=DEFAULT_CONFIG["ema_decay"])
    parser.add_argument("--num_workers", type=int, default=DEFAULT_CONFIG["num_workers"])
    parser.add_argument("--mixed_precision", choices=["bf16", "fp16", "none"], default=DEFAULT_CONFIG["mixed_precision"])

    parser.add_argument("--log_interval", type=int, default=DEFAULT_CONFIG["log_interval"])
    parser.add_argument("--sample_interval", type=int, default=DEFAULT_CONFIG["sample_interval"])
    parser.add_argument("--ckpt_interval", type=int, default=DEFAULT_CONFIG["ckpt_interval"])
    parser.add_argument("--n_samples_to_generate", type=int, default=DEFAULT_CONFIG["n_samples_to_generate"])
    parser.add_argument("--sample_lengths", type=parse_int_list, default=DEFAULT_CONFIG["sample_lengths"])
    parser.add_argument("--sample_ddim_steps", type=int, default=DEFAULT_CONFIG["sample_ddim_steps"])

    parser.add_argument("--out_dir", type=str, default=DEFAULT_CONFIG["out_dir"])
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    return parser.parse_args()


def main():
    args = parse_args()
    config = dict(DEFAULT_CONFIG)
    config.update(vars(args))
    if not config["sample_lengths"] or any(
        length <= 0 or length > config["target_length"] for length in config["sample_lengths"]
    ):
        raise ValueError("sample_lengths must be a non-empty list of lengths between 1 and target_length.")
    train(config)


if __name__ == "__main__":
    main()
