"""
GeoAMT-DiT 训练脚本
====================
功能:
    - 多 source 混合数据(real + syn_amt_A + syn_amt_B + syn_mt)
    - 实测 : 合成 = 1 : 1 加权采样
    - DDPM 训练(bf16 + grad clip + EMA)
    - 周期性 checkpoint + 采样可视化
    - 支持单卡 / 多卡(DDP)

用法(单卡):
    python train.py --data_root data/generator_sources --out_dir runs/exp_v1

用法(多卡): 在单机多卡场景下，一个进程对应一张 GPU。设置为 2，就代表你要使用当前机器上的 2 张 GPU。
    torchrun --nproc_per_node=2 train.py --data_root data/generator_sources --out_dir runs/exp_v1
    torchrun 的作用就是给这两个进程“牵线搭桥”，让它们知道：
    我们一共有几个兄弟（WORLD_SIZE=2）。
    我是老大还是老二（RANK=0 或 1）。
    我们该怎么通信（自动配置好通信后端）。

训练阶段：
干净样本 x_0
    ↓
随机时间步 t
    ↓
q_sample 加噪
    ↓
得到 x_t 和真实噪声 ε
    ↓
GeoAMT_DiT(x_t, t)
    ↓
预测噪声 ε_pred
    ↓
MSE(ε_pred, ε)
    ↓
更新模型


采样阶段：
随机噪声 x_T
    ↓
p_sample
    ↓
x_{T-1}
    ↓
p_sample
    ↓
x_{T-2}
    ↓
...
    ↓
x_0
    ↓
生成 AMT 信号
"""

import os
import sys
import json
import math
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
'''
3. 底层原理：分布式数据并行 (DDP)
当你用这个命令启动后，底层使用的是 DDP (DistributedDataParallel) 模式：
模型复制：2 张 GPU 上会各自加载一份完整的模型。
数据分片：你的 Dataset 数据会被自动切分成 2 份，GPU 0 处理前半部分，GPU 1 处理后半部分（Batch Size 会翻倍，比如代码里写 batch_size=16，实际总 batch_size 是 32）。
梯度同步：每次反向传播计算完梯度后，两张卡会通过 All-Reduce 算法互相交换并平均梯度，确保两张卡上的模型参数始终保持一致。
'''

# 让 Python 能找到上一级的 dataset 工具
sys.path.insert(0, str(Path(__file__).parent))


from model import GeoAMT_DiT
from diffusion import GaussianDiffusion
from ema import EMA

try:
    from geoamt_clean_dataset import MultiSourceCleanDataset, make_balanced_sampler
except ImportError as e:
    print(f"⚠️ 无法导入 geoamt_clean_dataset: {e}")
    print(f"   请确保 geoamt_preprocess_v2/geoamt_clean_dataset.py 在 Python path 中")
    raise


# ============================================================
# 配置(默认值,可通过命令行覆盖)
# ============================================================
DEFAULT_CONFIG = {
    # 数据
    'data_root': 'data/generator_sources',
    'use_sources': ['real', 'syn_amt_A', 'syn_amt_B', 'syn_mt'],
    'real_to_syn_ratio': 1.0,        # 实测 : 合成 权重比    合成数据的三个文件夹 每个取1/3
    'split_indices': None,           # 严格重训时使用 split_indices.npz 中的 *_train
    
    # 模型
    'in_channels': 4,
    'input_length': 2048,
    'patch_size': 8,
    'hidden_dim': 384,
    'depth': 12,
    'num_heads': 6,   # num_heads  → 多头自注意力分成几个头
    'mlp_ratio': 4.0, # mlp_ratio  → MLP 中间隐藏层放大多少倍
    
    # 扩散
    # cosine schedule 在生成质量上常常更稳定一些
    # linear schedule 更简单传统
    'num_timesteps': 1000,
    'beta_schedule': 'linear',       # 'linear' 或 'cosine'
    
    # 优化器
    'batch_size': 128,               # 单卡 batch size
    'lr': 1e-4,
    'weight_decay': 0.0,
    'adam_beta1': 0.9,
    'adam_beta2': 0.999,
    'grad_clip': 1.0,
    
    # 训练
    'num_steps': 200_000,            # 总训练步数
    'warmup_steps': 1000,            # 学习率 warmup
    'ema_decay': 0.9999,
    'mixed_precision': 'bf16',       # 'bf16', 'fp16', 或 'none'
    'num_workers': 4,
    
    # 日志 / 保存
    'log_interval': 50,
    'sample_interval': 5000,         # 每 5000 步采样一次
    'ckpt_interval': 10_000,         # 每 10000 步保存一次
    'n_samples_to_generate': 8,      # 每次采样生成几条
    
    # 输出
    'out_dir': 'runs/exp_v1',
    'resume_from': None,             # checkpoint 路径,继续训练  'resume_from': './outputs/geoamt_dit/ckpts/ckpt_step0050000.pt'
    
    # 随机种子
    'seed': 42,
}


# ============================================================
# 工具函数
# ============================================================
def setup_ddp():
    """初始化 DDP(如果通过 torchrun 启动)"""
    if 'RANK' not in os.environ:
        # 直接 python train.py，环境变量里没有 RANK，函数会直接返回 (0,1,0)。这样你调试时不需要改代码或注释分布式逻辑。
        return 0, 1, 0   # 单卡:rank=0, world_size=1, local_rank=0
    
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    '''
    torchrun 的作用就是给这两个进程“牵线搭桥”，让它们知道：
    我们一共有几个兄弟（WORLD_SIZE=2）。
    我是老大还是老二（RANK=0 或 1）。
    我们该怎么通信（自动配置好通信后端）。
    '''
    
    return rank, world_size, local_rank


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def get_lr(step, warmup_steps, total_steps, base_lr, min_lr_ratio=0.1):
    """Warmup + cosine decay"""
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    
    # cosine decay from base_lr to min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    min_lr = base_lr * min_lr_ratio
    return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ============================================================
# Dataset 构建
# ============================================================
def build_dataloader(config, world_size=1, rank=0):
    """构造混合多 source 的 DataLoader"""
    data_root = Path(config['data_root'])
    
    # 收集存在的 source 路径
    sources = {}
    for src in config['use_sources']:
        h5_path = data_root / src / 'clean_normalized.h5'   # 取归一化的数据
        if h5_path.exists():
            sources[src] = str(h5_path)
        else:
            if is_main_process():
                print(f"  ⚠️ 找不到 {h5_path},跳过 source '{src}'")
    
    if not sources:
        raise FileNotFoundError(f"在 {data_root} 下找不到任何 clean_normalized.h5")
    

    '''
    这个 MultiSourceCleanDataset 是核心。
    它内部不是直接把所有数据读进内存，而是 给每一个 source 单独创建一个 CleanAMTDataset。
    也就是说：
    MultiSourceCleanDataset
    ├── real       -> CleanAMTDataset(real/clean_normalized.h5)
    ├── syn_amt_A  -> CleanAMTDataset(syn_amt_A/clean_normalized.h5)
    ├── syn_amt_B  -> CleanAMTDataset(syn_amt_B/clean_normalized.h5)
    └── syn_mt     -> CleanAMTDataset(syn_mt/clean_normalized.h5)
    CleanAMTDataset 负责读取单个 HDF5 文件中的切片数据，返回的数据形状是：
    (4, 2048)
    MultiSourceCleanDataset 会建立一个全局索引映射。
    '''
    indices_by_source = {}
    split_indices_path = config.get('split_indices')
    if split_indices_path:
        split_indices_path = Path(split_indices_path)
        if not split_indices_path.exists():
            raise FileNotFoundError(f"split_indices not found: {split_indices_path}")
        with np.load(split_indices_path, allow_pickle=False) as split_file:
            for src in sources:
                key = f"{src}_train"
                if key not in split_file:
                    raise KeyError(f"{split_indices_path} missing key '{key}'")
                indices_by_source[src] = np.asarray(split_file[key], dtype=np.int64)
        if is_main_process():
            print(f"  Using strict train split: {split_indices_path}")

    dataset = MultiSourceCleanDataset(
        sources, augment=True,
        augment_config={'flip_prob': 0.5, 'swap_xy_prob': 0.5},  # 第一，时间反转   第二，交换 x/y 分量：
        indices_by_source=indices_by_source,
    )
    
    if is_main_process():
        info = dataset.get_info()
        print(f"\n  数据集信息:")
        print(f"    总切片: {info['total']:,}")
        for s, n in info['sources'].items():
            print(f"    {s:15s}: {n:>8,}")
    

    # 加权采样器
    # 实测占 ratio 份,合成所有 source 总共占 1 份(平均分给每个合成 source)
    ratio = config['real_to_syn_ratio']
    syn_sources = [s for s in sources if s != 'real']
    n_syn = len(syn_sources)   # 3 
    
    weight_per_source = {}
    for src in sources:
        if src == 'real':
            weight_per_source[src] = ratio
        else:
            weight_per_source[src] = 1.0 / n_syn if n_syn > 0 else 0.0
    """
    构造加权采样器,使不同 source 按指定比例采样。
    weight_per_source: dict, {source_name: weight}
        比如 {'real': 1.0, 'syn_amt_A': 1/3, 'syn_amt_B': 1/3, 'syn_mt': 1/3}
        means real 在最终 batch 里占 50%(1 / (1 + 1)),syn_* 三者合占另一半
    num_samples: 一个 epoch 取多少样本(默认 = 数据集总长)
    replacement: 是否带替换地采样(默认 True,加权采样必须 True)
    """
    # 返回一个 “按权重随机抽样的索引生成器”。
    # [100, 23, 5001, 5001, 12, 78, 3002, ...]
    sampler = make_balanced_sampler(
        dataset, weight_per_source,
        num_samples=len(dataset),
        replacement=True,
    )
    
    # 在 DDP 中传入不同 seed
    # sampler = make_balanced_sampler2(
    #     dataset,
    #     weight_per_source,
    #     num_samples=len(dataset),
    #     replacement=True,
    #     seed=config['seed'] + rank,
    # )

    # rank 0 用 seed = 42
    # rank 1 用 seed = 43
    # rank 2 用 seed = 44
    # rank 3 用 seed = 45

    if is_main_process():
        print(f"\n  采样权重:")
        for s, w in weight_per_source.items():
            print(f"    {s:15s}: {w:.4f}")
    

    # DDP 下:每张卡看不同子集
    # WeightedRandomSampler 和 DistributedSampler 不直接兼容
    # 简单做法:不同 rank 用不同 seed,DataLoader 自然分散
    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        sampler=sampler,  # WeightedRandomSampler(weights, num_samples=num_samples, replacement=replacement)
        num_workers=config['num_workers'],
        pin_memory=True,  # 表示把 CPU 内存固定住，方便更快地传到 GPU。
        drop_last=True,   # 表示如果最后剩下的数据不够一个完整 batch，就丢掉。
        persistent_workers=config['num_workers'] > 0,  # 表示如果 num_workers > 0，那么每个 epoch 结束后，worker 进程不销毁，下一个 epoch 继续用。
    )
    
    return dataset, loader


# ============================================================
# 采样可视化
# ============================================================
@torch.no_grad()
def sample_and_save(ema, diffusion, config, step, out_dir, device):
    """用 EMA 模型生成几个样本并保存"""
    ema.shadow_model.eval()
    
    shape = (config['n_samples_to_generate'], 
             config['in_channels'], 
             config['input_length'])
    
    # 用 DDIM 加速(50 步)
    samples = diffusion.ddim_sample(
        ema.shadow_model, shape, 
        num_steps=50, eta=0.0, device=device, progress=False
    )
    
    samples_np = samples.cpu().numpy()
    
    # 保存为 .npy
    save_dir = Path(out_dir) / 'samples'
    save_dir.mkdir(exist_ok=True, parents=True)
    np.save(save_dir / f'samples_step{step:07d}.npy', samples_np)
    
    # 尝试画图(如果有 matplotlib)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        n = min(4, samples_np.shape[0])
        fig, axes = plt.subplots(n, 4, figsize=(20, n * 2.5), sharex=True)
        if n == 1:
            axes = axes[np.newaxis, :]
        
        chan_names = ['Ex', 'Ey', 'Hx', 'Hy']
        chan_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i in range(n):
            for ch in range(4):
                axes[i, ch].plot(samples_np[i, ch], 
                                color=chan_colors[ch], linewidth=0.6)
                axes[i, ch].grid(alpha=0.3)
                if i == 0:
                    axes[i, ch].set_title(chan_names[ch])
                if ch == 0:
                    axes[i, ch].set_ylabel(f'Sample {i}')
        
        fig.suptitle(f'EMA samples @ step {step}')
        fig.tight_layout()
        fig.savefig(save_dir / f'samples_step{step:07d}.png', dpi=80)
        plt.close(fig)
    except ImportError:
        pass
    
    return samples_np


# ============================================================
# Checkpoint
# ============================================================
def save_checkpoint(path, step, model, ema, optimizer, scaler, config):
    """保存训练状态"""
    if not is_main_process():
        return
    
    raw_model = model.module if isinstance(model, DDP) else model
    
    ckpt = {
        'step': step,
        'model': raw_model.state_dict(),
        'ema': ema.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': config,
    }
    if scaler is not None:
        ckpt['scaler'] = scaler.state_dict()
    
    torch.save(ckpt, path)
    print(f"  💾 保存 checkpoint: {path}")


def load_checkpoint(path, model, ema, optimizer, scaler):
    """加载训练状态"""
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.load_state_dict(ckpt['model'])
    ema.load_state_dict(ckpt['ema'])
    optimizer.load_state_dict(ckpt['optimizer'])
    if scaler is not None and 'scaler' in ckpt:
        scaler.load_state_dict(ckpt['scaler'])
    
    return ckpt['step']


# ============================================================
# 主训练循环
# ============================================================
def train(config):
    # ========== DDP 初始化 ==========
    rank, world_size, local_rank = setup_ddp()
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')
    
    if is_main_process():
        print("=" * 60)
        print(f"GeoAMT-DiT 训练")
        print("=" * 60)
        print(f"  Device: {device}")
        print(f"  World size: {world_size}")
        print(f"  Mixed precision: {config['mixed_precision']}")
        print(f"  Batch size (per device): {config['batch_size']}")
        if world_size > 1:
            print(f"  Global batch size: {config['batch_size'] * world_size}")
    
    # 设置随机种子
    seed = config['seed'] + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # ========== 输出目录 ==========
    out_dir = Path(config['out_dir'])
    if is_main_process():
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'ckpts').mkdir(exist_ok=True)
        (out_dir / 'samples').mkdir(exist_ok=True)
        
        # 保存配置
        with open(out_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)



    # 存储训练日志
    loss_log_path = out_dir / "loss_history.csv"

    # ========== 数据 ==========
    if is_main_process():
        print(f"\n[加载数据]")
    dataset, loader = build_dataloader(config, world_size, rank)
    

    # ========== 模型 ==========
    if is_main_process():
        print(f"\n[构建模型]")
    
    model = GeoAMT_DiT(
        in_channels=config['in_channels'],
        input_length=config['input_length'],
        patch_size=config['patch_size'],
        hidden_dim=config['hidden_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=config['mlp_ratio'],
    ).to(device)
    
    # 为了 DDP 多卡训练
    if is_main_process():
        print(f"  参数量: {count_params(model) / 1e6:.2f} M")
    
    # 如果是多卡训练，就把模型包装成 DDP。
    # DDP 包装
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    # EMA(用 raw model 初始化)
    raw_model = model.module if isinstance(model, DDP) else model
    # EMA 模型不参与梯度训练，只在每次 optimizer.step() 后用“指数滑动平均”的方式缓慢跟随主模型
    # 扩散模型采样时通常使用 EMA 模型，因为它比主模型当前权重更平滑、更稳定，生成质量通常更好。

    # EMA 是主模型的平滑版本。主模型每一步都被 optimizer 更新，变化比较快；
    # EMA 模型不会反向传播，只是每一步慢慢跟随主模型：EMA参数 = decay × EMA旧参数 + (1 - decay) × 当前主模型参数
    ema = EMA(raw_model, decay=config['ema_decay']).to(device)
    

    # GaussianDiffusion：负责加噪、去噪流程
    # GeoAMT_DiT：负责预测当前噪声

    # ========== 扩散调度器 ==========
    # 这个不是神经网络，而是扩散过程控制器。
    # 1. 训练时生成 x_t
    # 2. 计算 loss
    # 3. 采样时从 x_T 反推到 x_0
    diffusion = GaussianDiffusion(
        num_timesteps=config['num_timesteps'],
        schedule=config['beta_schedule'],
        device=device,
    )
    

    # ========== 优化器 ==========
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        betas=(config['adam_beta1'], config['adam_beta2']),
        weight_decay=config['weight_decay'],
    )
    
    # 混合精度
    # mixed_precision = 'none' → 普通 FP32 训练
    # mixed_precision = 'fp16' → FP16 混合精度，需要 GradScaler
    # mixed_precision = 'bf16' → BF16 混合精度，一般不需要 GradScaler
    use_amp = config['mixed_precision'] != 'none'
    amp_dtype = torch.bfloat16 if config['mixed_precision'] == 'bf16' else torch.float16
    scaler = torch.cuda.amp.GradScaler() if config['mixed_precision'] == 'fp16' else None
    
    # ========== 加载 checkpoint(如果有) ==========
    start_step = 0
    if config['resume_from']:
        if is_main_process():
            print(f"\n  恢复训练: {config['resume_from']}")
        start_step = load_checkpoint(
            config['resume_from'], model, ema, optimizer, scaler
        )
        if is_main_process():
            print(f"  从 step {start_step} 继续")

    if is_main_process() and (start_step == 0 or not loss_log_path.exists()):
        with open(loss_log_path, "w", encoding="utf-8") as f:
            f.write("step,loss,avg_loss,lr,steps_per_sec\n")
    
    # ========== 训练循环 ==========
    if is_main_process():
        print(f"\n[开始训练]")
        print(f"  总步数: {config['num_steps']:,}")
        print(f"  Warmup: {config['warmup_steps']:,}")
    
    model.train()
    step = start_step
    loss_window = []
    last_log_time = time.time()
    
    data_iter = iter(loader)
    
    while step < config['num_steps']:  # 这里不是按 epoch 训练，而是按 step 训练。
        # 取一个 batch(loader 用完就重启)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        
        x_0 = batch.to(device, non_blocking=True)  # 干净的样本
        
        # 调整学习率
        cur_lr = get_lr(step, config['warmup_steps'], config['num_steps'], config['lr'])
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr
        
        # 前向 + 反向
        # 每次反向传播前要清空上一轮梯度。
        # set_to_none=True 比直接置零更省显存、速度更快一些。
        optimizer.zero_grad(set_to_none=True)
        
        if use_amp:
            with torch.autocast(device_type='cuda', dtype=amp_dtype):
                loss = diffusion.training_loss(model, x_0)
            
            if scaler is not None:   # fp16 路径
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])  # 梯度裁剪  防止梯度爆炸
                scaler.step(optimizer)
                scaler.update()
            else:                     # bf16 路径
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                optimizer.step()
        else:
            loss = diffusion.training_loss(model, x_0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()
        
        # EMA 更新
        ema.update(raw_model)
        

        step += 1
        loss_window.append(loss.item())
        
        # ========== 日志 ==========
        if step % config['log_interval'] == 0 and is_main_process():
            avg_loss = sum(loss_window) / len(loss_window)
            loss_window = loss_window[-100:]   # 只保留最近 100 个
            
            elapsed = time.time() - last_log_time
            steps_per_sec = config['log_interval'] / max(elapsed, 1e-6)
            last_log_time = time.time()
            
            print(f"  step {step:>7,} | loss {avg_loss:.4f} | "
                  f"lr {cur_lr:.2e} | {steps_per_sec:.2f} step/s")
            
            with open(loss_log_path, "a", encoding="utf-8") as f:
                f.write(f"{step},{loss.item():.6f},{avg_loss:.6f},{cur_lr:.8e},{steps_per_sec:.4f}\n")
        


        # ========== 周期采样 ==========
        if step % config['sample_interval'] == 0 and is_main_process():
            print(f"  🎨 生成样本(step {step})...")
            sample_and_save(ema, diffusion, config, step, out_dir, device)
        


        # ========== Checkpoint ==========
        if step % config['ckpt_interval'] == 0 and is_main_process():
            ckpt_path = out_dir / 'ckpts' / f'ckpt_step{step:07d}.pt'
            save_checkpoint(ckpt_path, step, model, ema, optimizer, scaler, config)
            
            # 同时保存一个 'latest.pt' 链接
            latest_path = out_dir / 'ckpts' / 'latest.pt'
            torch.save({
                'step': step,
                'model': raw_model.state_dict(),
                'ema': ema.state_dict(),
                'config': config,
            }, latest_path)
    

    # 训练结束
    if is_main_process():
        print(f"\n[训练完成]")
        save_checkpoint(out_dir / 'ckpts' / 'final.pt', step, model, ema, optimizer, scaler, config)
    
    if world_size > 1:
        dist.destroy_process_group()


# ============================================================
# Argparse + 启动
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description='GeoAMT-DiT 训练')
    
    # 数据
    p.add_argument('--data_root', type=str, default=DEFAULT_CONFIG['data_root'])
    p.add_argument('--use_sources', type=str, nargs='+', 
                   default=DEFAULT_CONFIG['use_sources'])
    p.add_argument('--real_to_syn_ratio', type=float, 
                   default=DEFAULT_CONFIG['real_to_syn_ratio'])
    p.add_argument(
        '--split_indices',
        type=str,
        default=DEFAULT_CONFIG['split_indices'],
        help='build_geoamt_test_split.py 生成的 split_indices.npz',
    )
    
    # 模型
    p.add_argument('--hidden_dim', type=int, default=DEFAULT_CONFIG['hidden_dim'])
    p.add_argument('--depth', type=int, default=DEFAULT_CONFIG['depth'])
    p.add_argument('--num_heads', type=int, default=DEFAULT_CONFIG['num_heads'])
    p.add_argument('--patch_size', type=int, default=DEFAULT_CONFIG['patch_size'])
    
    # 扩散
    p.add_argument('--num_timesteps', type=int, default=DEFAULT_CONFIG['num_timesteps'])
    p.add_argument('--beta_schedule', type=str, default=DEFAULT_CONFIG['beta_schedule'])
    
    # 训练
    p.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'])
    p.add_argument('--lr', type=float, default=DEFAULT_CONFIG['lr'])
    p.add_argument('--num_steps', type=int, default=DEFAULT_CONFIG['num_steps'])
    p.add_argument('--warmup_steps', type=int, default=DEFAULT_CONFIG['warmup_steps'])
    p.add_argument('--ema_decay', type=float, default=DEFAULT_CONFIG['ema_decay'])
    p.add_argument('--num_workers', type=int, default=DEFAULT_CONFIG['num_workers'])
    p.add_argument('--mixed_precision', type=str, 
                   choices=['bf16', 'fp16', 'none'],
                   default=DEFAULT_CONFIG['mixed_precision'])
    
    # 日志
    p.add_argument('--log_interval', type=int, default=DEFAULT_CONFIG['log_interval'])
    p.add_argument('--sample_interval', type=int, default=DEFAULT_CONFIG['sample_interval'])
    p.add_argument('--ckpt_interval', type=int, default=DEFAULT_CONFIG['ckpt_interval'])
    
    # 其他
    p.add_argument('--out_dir', type=str, default=DEFAULT_CONFIG['out_dir'])
    p.add_argument('--resume_from', type=str, default=None)
    p.add_argument('--seed', type=int, default=DEFAULT_CONFIG['seed'])
    
    return p.parse_args()


def main():
    args = parse_args()
    
    # 合并到 config
    config = dict(DEFAULT_CONFIG)
    config.update(vars(args))
    
    train(config)


if __name__ == '__main__':
    main()
