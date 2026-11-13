"""
Train a conditional Noise-DiT on real + synthetic AMT noise clips.

Example:
    python train_noise.py ^
      --data_root noise_data ^
      --synthetic_root noise_data/Synthetic-Norm ^
      --out_dir runs/noise_dit_all_lengths ^
      --batch_size 128

输入输出: (B, 1, 2048)
条件: 噪声类别 type_id + 有效长度 length_ratio
可选条件: source_id, 也就是 real/synthetic

python train_noise.py 
--data_root noise_data 
--synthetic_root noise_data/Synthetic-Norm 
--out_dir runs/noise_dit_all_lengths 
--batch_size 128 
--num_steps 200000


当前训练数据实际是:
实测:
    Real-Len100 ~ Real-Len1500
    Real-2048

合成:
    Synthetic-Norm\Synthetic-Len*


只训练实测:
python train_noise.py --data_root noise_data --no_synthetic --out_dir runs/noise_real_only

只训练合成：
python train_noise.py --data_root noise_data --synthetic_root noise_data/Synthetic-Norm --no_real --out_dir runs/noise_syn_only
"""

from __future__ import annotations

#######
"""
Train a conditional Noise-DiT on real + synthetic AMT noise clips.

Example:
    python train_noise.py ^
      --data_root noise_data ^
      --synthetic_root noise_data/Synthetic-Norm ^
      --out_dir runs/noise_dit_all_lengths ^
      --batch_size 128
python train_noise.py --data_root noise_data --synthetic_root noise_data/Synthetic-Norm --out_dir runs/noise_dit_all_lengths


整体工作流程：
读取 real / synthetic 噪声数据
        ↓
使用所有不超过 target_length 的长度，也可通过 train_lengths 筛选
        ↓
构建 DataLoader 和 WeightedRandomSampler
        ↓
构建 ConditionalNoiseDiT 网络
        ↓
构建 GaussianDiffusion 扩散过程
        ↓
训练模型预测扩散噪声
        ↓
定期采样、保存 checkpoint、记录 loss
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from diffusion import GaussianDiffusion
from ema import EMA
from noise_dataset import NOISE_TYPE_NAMES, NoiseNPYDataset, make_noise_sampler
from noise_model import ConditionalNoiseDiT


DEFAULT_CONFIG = {
    # Data
    "data_root": r"noise_data",
    "synthetic_root": r"noise_data/Synthetic-Norm",
    "include_real": True,
    "include_synthetic": True,

    "fixed_length": None,
    "target_length": 2048,
    "train_lengths": None,  # None: 使用所有不超过 target_length 的原始长度。

    "data_normalize": "rms",

    "random_embed": True,  # 将短片段随机放入 target_length 长度的零填充序列。

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
    "condition_on_length": True,  # 使用有效长度占比 length_ratio 作为模型条件。
    "condition_on_source": False,

    # Diffusion
    "num_timesteps": 1000,
    "beta_schedule": "linear",

    # Optimizer/training
    "batch_size": 128,
    "lr": 1e-4,
    "weight_decay": 0.0,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "grad_clip": 1.0,
    "num_steps": 200_000,
    "warmup_steps": 1000,
    "ema_decay": 0.9999,

    # bf16  使用 bfloat16 混合精度  比 float32 省显存  训练速度通常更快  数值范围接近 float32  比 fp16 稳定  一般不需要 GradScaler
    # fp16  使用 float16 混合精度
    # none  不使用混合精度，全程 float32  数值最稳定  不容易 Nan 适合调试代码  loss 更容易判断是否正常
    "mixed_precision": "bf16",  # fp16 none

    "num_workers": 4,

    # Logging/saving
    "log_interval": 50,
    "sample_interval": 5000,
    "ckpt_interval": 10_000,
    "n_samples_to_generate": 20,
    "sample_lengths": [100, 300, 750, 1500, 2048],
    "sample_ddim_steps": 50,

    # Output
    "out_dir": "runs/noise_dit_all_lengths",
    "resume_from": None,
    "seed": 42,
}


def parse_int_list(text):
    if text is None:
        return None
    if isinstance(text, (list, tuple)):
        return [int(value) for value in text]

    if str(text).strip().lower() == "all":
        return None
    
    values = [part.strip() for part in str(text).split(",") if part.strip()]
    return [int(value) for value in values]


def get_lr(step, warmup_steps, total_steps, base_lr, min_lr_ratio=0.1):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    min_lr = base_lr * min_lr_ratio
    return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def build_dataloader(config):
    # 数据集构建
    dataset = NoiseNPYDataset(
        data_root=config["data_root"],
        synthetic_root=config["synthetic_root"],
        target_length=config["target_length"],
        include_real=config["include_real"],
        include_synthetic=config["include_synthetic"],
        max_real_original_len=config["target_length"],

        selected_lengths=config["train_lengths"],  # None 表示不额外筛选原始长度。

        normalize=config["data_normalize"],
        random_embed=config["random_embed"],
        flip_prob=config["flip_prob"],
        sign_flip_prob=config["sign_flip_prob"],
    )
    # 采样权重设置
    # 训练时尽量平衡不同噪声类型，同时尽量平衡 real 和 synthetic 数据来源。
    # 如果：--no_real  那么数据集中只有 synthetic，real_weight 和 synthetic_weight 基本没有实际意义；
    # 此时主要靠 balance_type=True 平衡 10 类噪声。
    # 如果你同时使用 real 和 synthetic，而 real 只有少数几类，synthetic 有 10 类，那么建议：--synthetic_weight 3.0
    sampler = make_noise_sampler(
        dataset,
        balance_type=config["balance_type"],
        balance_source=config["balance_source"],
        real_weight=config["real_weight"],
        synthetic_weight=config["synthetic_weight"],
        num_samples=len(dataset),
    )
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        sampler=sampler,
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
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
    ckpt = {
        "step": step,
        "model": model.state_dict(),
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
    model.load_state_dict(ckpt["model"])
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
        # 交错长度，使默认 20 条预览覆盖全部类型和代表长度。
        length_index = (i + i // config["num_noise_types"]) % len(sample_lengths)
        length_values.append(sample_lengths[length_index])

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ckpts").mkdir(exist_ok=True)
    (out_dir / "samples").mkdir(exist_ok=True)

    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


    '''  数据集的构建  '''
    dataset, loader = build_dataloader(config)
    info = dataset.get_info()
    with (out_dir / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)



    print("=" * 70)
    print("Conditional Noise-DiT training")
    print("=" * 70)
    print(f"device: {device}")
    print(f"samples: {len(dataset):,}")
    print(f"type distribution: {info['type_distribution']}")
    print(f"source distribution: {info['source_distribution']}")
    print(f"length distribution: {info['length_distribution']}")

    model = build_model(config, device)
    print(f"parameters: {count_params(model) / 1e6:.2f} M")

    ema = EMA(model, decay=config["ema_decay"])

    '''  扩散模型部分  
    默认：
    num_timesteps = 1000
    beta_schedule = "linear"
    '''
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

    # 是否使用混合精度训练
    use_amp = config["mixed_precision"] != "none" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if config["mixed_precision"] == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler() if use_amp and config["mixed_precision"] == "fp16" else None


    start_step = 0
    if config["resume_from"]:
        print(f"resume from: {config['resume_from']}")
        start_step = load_checkpoint(config["resume_from"], model, ema, optimizer, scaler, device)
        print(f"resumed step: {start_step}")

    loss_log_path = out_dir / "loss_log.csv"
    if start_step == 0 or not loss_log_path.exists():
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

        ema.update(model)
        step += 1
        loss_window.append(float(loss.item()))

        if step % config["log_interval"] == 0:
            avg_loss = sum(loss_window[-100:]) / min(len(loss_window), 100)
            elapsed = time.time() - last_log_time
            steps_per_sec = config["log_interval"] / max(elapsed, 1e-6)
            last_log_time = time.time()
            print(
                f"step {step:>7,} | loss {loss.item():.4f} | "
                f"avg {avg_loss:.4f} | lr {cur_lr:.2e} | {steps_per_sec:.2f} step/s"
            )
            with loss_log_path.open("a", encoding="utf-8") as f:
                f.write(f"{step},{loss.item():.6f},{avg_loss:.6f},{cur_lr:.8e},{steps_per_sec:.4f}\n")

        if step % config["sample_interval"] == 0:
            print(f"sampling @ step {step} ...")
            sample_and_save(ema, diffusion, config, step, out_dir, device)
            model.train()

        if step % config["ckpt_interval"] == 0:
            ckpt_path = out_dir / "ckpts" / f"ckpt_step{step:07d}.pt"
            save_checkpoint(ckpt_path, step, model, ema, optimizer, scaler, config)
            latest_path = out_dir / "ckpts" / "latest.pt"
            save_checkpoint(latest_path, step, model, ema, optimizer, scaler, config)

    save_checkpoint(out_dir / "ckpts" / "final.pt", step, model, ema, optimizer, scaler, config)
    print("training complete")


def parse_args():
    parser = argparse.ArgumentParser(description="Train conditional Noise-DiT")

    parser.add_argument("--data_root", type=str, default=DEFAULT_CONFIG["data_root"])
    parser.add_argument("--synthetic_root", type=str, default=DEFAULT_CONFIG["synthetic_root"])
    parser.add_argument("--include_real", action="store_true", default=DEFAULT_CONFIG["include_real"])
    parser.add_argument("--no_real", dest="include_real", action="store_false")
    parser.add_argument("--include_synthetic", action="store_true", default=DEFAULT_CONFIG["include_synthetic"])
    parser.add_argument("--no_synthetic", dest="include_synthetic", action="store_false")
    parser.add_argument(
        "--fixed_length",
        type=int,
        default=DEFAULT_CONFIG["fixed_length"],
        help="Use one original length only and set target/input/sample lengths together.",
    )
    parser.add_argument("--target_length", type=int, default=DEFAULT_CONFIG["target_length"])
    parser.add_argument("--input_length", type=int, default=DEFAULT_CONFIG["input_length"])
    parser.add_argument(
        "--train_lengths",
        type=parse_int_list,
        default=DEFAULT_CONFIG["train_lengths"],
        help="Original lengths: 'all' (default) uses every available length <= target_length; "
        "or a comma-separated list, e.g. 100 or 150,300.",
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

    if config["fixed_length"] is not None:
        fixed_length = int(config["fixed_length"])
        config["target_length"] = fixed_length
        config["input_length"] = fixed_length
        config["train_lengths"] = [fixed_length]
        config["sample_lengths"] = [fixed_length]

    config["train_lengths"] = parse_int_list(config["train_lengths"])
    config["sample_lengths"] = parse_int_list(config["sample_lengths"]) or [config["target_length"]]

    if config["input_length"] != config["target_length"]:
        raise ValueError(
            "input_length must equal the padded target_length. "
            "For example, use --fixed_length 2048 for the 2048-only model."
        )
    if config["train_lengths"] is not None:
        if not config["train_lengths"]:
            raise ValueError("train_lengths cannot be empty.")
        too_long = [length for length in config["train_lengths"] if length > config["target_length"]]
        if too_long:
            raise ValueError(
                f"train_lengths must be <= target_length. Got {too_long}, "
                f"target_length={config['target_length']}."
            )
        if not config["condition_on_length"] and len(set(config["train_lengths"])) > 1:
            print("warning: multiple train_lengths but condition_on_length=False")
    train(config)


if __name__ == "__main__":
    main()




'''
多长度噪声统一嵌入 2048 并用 length_ratio 控制的版本
'''
