"""
GeoAMT-DiT 采样脚本
====================
从训练好的 checkpoint 加载,生成 N 个新的 (4, 2048) 样本

用法:
    python sample.py --ckpt runs/exp_v1/ckpts/latest.pt --n 1000
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import h5py

sys.path.insert(0, str(Path(__file__).parent))
from model import GeoAMT_DiT
from diffusion import GaussianDiffusion


def load_model_from_ckpt(ckpt_path, use_ema=True, device='cuda'):
    """加载 checkpoint,返回 model + diffusion + config"""
    print(f"  加载 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    config = ckpt['config']
    
    model = GeoAMT_DiT(
        in_channels=config['in_channels'],
        input_length=config['input_length'],
        patch_size=config['patch_size'],
        hidden_dim=config['hidden_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=config['mlp_ratio'],
    ).to(device)
    
    # 优先用 EMA 权重(质量更好)
    if use_ema and 'ema' in ckpt:
        model.load_state_dict(ckpt['ema'])
        print(f"  ✓ 加载 EMA 权重")
    else:
        model.load_state_dict(ckpt['model'])
        print(f"  ✓ 加载主模型权重")
    
    model.eval()
    
    diffusion = GaussianDiffusion(
        num_timesteps=config['num_timesteps'],
        schedule=config['beta_schedule'],
        device=device,
    )
    
    return model, diffusion, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='checkpoint 路径')
    parser.add_argument('--n', type=int, default=1000, help='生成样本数')
    parser.add_argument('--batch_size', type=int, default=64, help='采样 batch size')

    parser.add_argument('--num_steps', type=int, default=50, 
                        help='DDIM 采样步数(50 一般够,1000 = 全 DDPM)')

    parser.add_argument('--out_path', type=str, default='samples.h5',
                        help='输出文件(.h5 或 .npy)')
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='使用 EMA 权重(默认 True)')
    parser.add_argument('--no_ema', dest='use_ema', action='store_false')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # 加载模型
    model, diffusion, config = load_model_from_ckpt(
        args.ckpt, use_ema=args.use_ema, device=device
    )
    
    print(f"\n  采样配置:")
    print(f"    样本数: {args.n}")
    print(f"    Batch size: {args.batch_size}")
    print(f"    DDIM 步数: {args.num_steps}")
    print(f"    输入 shape: ({config['in_channels']}, {config['input_length']})")
    
    # 分 batch 采样
    all_samples = []
    n_batches = (args.n + args.batch_size - 1) // args.batch_size
    
    for bi in range(n_batches):
        bs = min(args.batch_size, args.n - bi * args.batch_size)
        shape = (bs, config['in_channels'], config['input_length'])
        
        print(f"\n  Batch {bi+1}/{n_batches} ({bs} 个样本)...")
        
        with torch.no_grad():
            if args.num_steps >= config['num_timesteps']:
                # 全步 DDPM
                samples = diffusion.sample(model, shape, device=device, progress=True)
            else:
                # DDIM 加速  # DDIM 采样(可选,采样步数少 → 更快)  它不是每一步都走，而是从 1000 个时间步里均匀挑 50 个：
                samples = diffusion.ddim_sample(
                    model, shape, num_steps=args.num_steps, eta=0.0,
                    device=device, progress=True
                )
        
        all_samples.append(samples.cpu().numpy())
    
    all_samples = np.concatenate(all_samples, axis=0)
    print(f"\n  生成完成: {all_samples.shape}")
    
    # 保存
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if out_path.suffix == '.npy':
        np.save(out_path, all_samples)
    else:
        with h5py.File(out_path, 'w') as f:
            f.create_dataset('data', data=all_samples.astype(np.float32),
                             compression='gzip', compression_opts=4)
            f.attrs['n_samples'] = len(all_samples)
            f.attrs['checkpoint_name'] = Path(args.ckpt).name
            f.attrs['num_steps'] = args.num_steps
            f.attrs['use_ema'] = args.use_ema
            f.attrs['config'] = json.dumps(config)
    
    print(f"  ✓ 已保存: {out_path}")
    print(f"  文件大小: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")
    
    # 简单统计
    print(f"\n  样本统计:")
    for ch, name in enumerate(['Ex', 'Ey', 'Hx', 'Hy']):
        v = all_samples[:, ch].flatten()
        print(f"    {name}: mean={v.mean():.3f}, std={v.std():.3f}, "
              f"min={v.min():.3f}, max={v.max():.3f}")


if __name__ == '__main__':
    main()
