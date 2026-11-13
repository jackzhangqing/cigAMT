"""
DDPM Gaussian Diffusion Scheduler
==================================
负责:
    - 计算 forward diffusion 的 noise 系数(α_t, β_t)
    - 给样本加噪(forward process)
    - 训练时:loss = MSE(model(x_t, t), ε)
    - 采样时:从 x_T 一步步去噪到 x_0

这段代码实现的是一个 DDPM 扩散模型调度器，也就是负责：
    训练时：给干净数据加噪，让模型学习预测噪声。
    生成时：从纯噪声开始，一步一步去噪，生成新的 AMT 信号。
它本身不是神经网络模型，而是扩散过程的“数学控制器”。
"""

import math
import torch
import torch.nn.functional as F
import numpy as np


def linear_beta_schedule(num_timesteps, beta_start=1e-4, beta_end=0.02):
    """DDPM 原始 schedule"""
    return torch.linspace(beta_start, beta_end, num_timesteps)


def cosine_beta_schedule(num_timesteps, s=0.008):
    """改进的 cosine schedule(iDDPM)"""
    steps = num_timesteps + 1
    t = torch.linspace(0, num_timesteps, steps) / num_timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class GaussianDiffusion:
    """DDPM 调度器"""
    
    def __init__(self, num_timesteps=1000, schedule='linear', device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device
        
        if schedule == 'linear':
            self.betas = linear_beta_schedule(num_timesteps).to(device)
        elif schedule == 'cosine':
            self.betas = cosine_beta_schedule(num_timesteps).to(device)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        # 预计算各种系数
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # 也就是从第 0 步到第 t 步累计保留了多少原始信号。
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # forward 采样系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # reverse 采样系数(后验均值/方差)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) 
            / (1.0 - self.alphas_cumprod)
        )
    
    @staticmethod
    def _extract(arr, t, x_shape):
        """从 (T,) 数组里按 t 取值,reshape 成可广播到 x_shape 的形式"""
        B = t.shape[0]
        out = arr.gather(-1, t)
        return out.reshape(B, *([1] * (len(x_shape) - 1)))
    
    # =========================================================
    # Forward diffusion: 从 x_0 加噪到 x_t
    # =========================================================
    def q_sample(self, x_0, t, noise=None):
        """
        x_0: (B, 4, L) 干净样本
        t:   (B,) 时间步
        noise: 可选,自定义噪声;None 则随机生成
        
        return: x_t, noise
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        x_t = sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise

        # x_t   ：加噪后的样本
        # noise ：真实加入的噪声
        return x_t, noise
    
    # =========================================================
    # 训练:计算 loss
    # =========================================================
    def training_loss(self, model, x_0, model_kwargs=None, **kwargs):
        """
        x_0: (B, 4, L) 干净样本
        return: scalar loss
        """
        B = x_0.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=x_0.device)  # 每个样本随机选一个扩散时间步。
        
        x_t, noise = self.q_sample(x_0, t)  # q_sample: 给 x_0 加噪得到 x_t,并返回真实噪声 noise
        if model_kwargs is None:
            model_kwargs = {}
        model_kwargs.update(kwargs)
        predicted_noise = model(x_t, t, **model_kwargs)
        
        loss = F.mse_loss(predicted_noise, noise)
        return loss
    
    # =========================================================
    # Reverse diffusion: 从 x_T 采样回 x_0
    # =========================================================
    @torch.no_grad()
    def p_sample(self, model, x_t, t, model_kwargs=None, **kwargs):
        """
        从 x_t 采样 x_{t-1}
        t: (B,) tensor
        """
        # 预测噪声
        if model_kwargs is None:
            model_kwargs = {}
        model_kwargs.update(kwargs)
        predicted_noise = model(x_t, t, **model_kwargs)
        
        # 用预测噪声反推 x_0
        sqrt_alpha_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        
        # x_0_pred = (x_t - sqrt(1-α_bar) * ε) / sqrt(α_bar)
        x_0_pred = (x_t - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
        x_0_pred = torch.clamp(x_0_pred, -5.0, 5.0)   # 归一化数据应该在 ~O(1)
        
        # 后验均值
        coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        # q(x_{t-1} | x_t, x_0)  上一时刻 x_{t-1} 应该由“预测的干净样本 x_0_pred”和“当前样本 x_t”共同决定。
        posterior_mean = coef1 * x_0_pred + coef2 * x_t
        
        # 后验方差
        posterior_log_var = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        # t=0 时不再加噪
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().reshape(-1, *([1] * (len(x_t.shape) - 1)))
        
        return posterior_mean + nonzero_mask * torch.exp(0.5 * posterior_log_var) * noise
    

    
    @torch.no_grad()
    def sample(self, model, shape, device=None, progress=True, model_kwargs=None, **kwargs):  # 这个函数用于生成新数据。
        """
        从纯噪声采样 N 个样本
        shape: (N, 4, L)

        所以 sample() 是标准 DDPM 采样，质量比较稳定，但需要走 1000 步，速度较慢。
        """
        if device is None:
            device = self.device
        
        x = torch.randn(shape, device=device)  # 第一步: 一开始完全是随机噪声。
        
        timesteps = list(range(self.num_timesteps))[::-1]  # 第二步：从 T-1 到 0 逐步去噪
        if progress:
            try:
                from tqdm import tqdm
                timesteps = tqdm(timesteps, desc='Sampling', leave=False)
            except ImportError:
                pass
        
        for t_int in timesteps:
            t = torch.full((shape[0],), t_int, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, model_kwargs=model_kwargs, **kwargs)   # 最终得到的x_0, 就生成的干净的AMT信号
        
        return x
    

    # =========================================================
    # DDIM 采样(可选,采样步数少 → 更快)  它不是每一步都走，而是从 1000 个时间步里均匀挑 50 个：
    # =========================================================
    @torch.no_grad()
    def ddim_sample(self, model, shape, num_steps=50, eta=0.0, device=None, progress=True, model_kwargs=None, **kwargs):
        """
        DDIM:从 T 步压缩到 num_steps 步,采样更快
        eta=0: deterministic DDIM
        eta=1: 等价于 DDPM
        """
        if device is None:
            device = self.device
        
        x = torch.randn(shape, device=device)
        
        # 选 num_steps 个时间步(均匀)
        step_indices = np.linspace(0, self.num_timesteps - 1, num_steps + 1).astype(int)[::-1]
        
        iterator = list(zip(step_indices[:-1], step_indices[1:]))
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc='DDIM sampling', leave=False)
            except ImportError:
                pass
        
        for t_int, t_prev_int in iterator:
            t = torch.full((shape[0],), t_int, device=device, dtype=torch.long)
            
            # 预测噪声
            if model_kwargs is None:
                model_kwargs = {}
            model_kwargs.update(kwargs)
            predicted_noise = model(x, t, **model_kwargs)
            
            alpha_t = self.alphas_cumprod[t_int]
            alpha_prev = self.alphas_cumprod[t_prev_int] if t_prev_int >= 0 else torch.tensor(1.0, device=device)
            
            # 估计 x_0
            x_0_pred = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            x_0_pred = torch.clamp(x_0_pred, -5.0, 5.0)  # 默认训练数据是归一化后的，数值大概在 O(1)。
            
            # DDIM 公式
            sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_prev)
            noise_term = torch.sqrt(1 - alpha_prev - sigma ** 2) * predicted_noise
            
            x = torch.sqrt(alpha_prev) * x_0_pred + noise_term
            if eta > 0 and t_prev_int > 0:
                x = x + sigma * torch.randn_like(x)
        
        return x


# ============================================================
# 快速测试
# ============================================================
if __name__ == '__main__':
    diffusion = GaussianDiffusion(num_timesteps=1000, device='cpu')
    
    # 测试 forward
    x_0 = torch.randn(4, 4, 2048)
    t = torch.tensor([0, 250, 500, 999])
    x_t, noise = diffusion.q_sample(x_0, t)
    print(f"x_0: {x_0.shape}, x_t: {x_t.shape}, noise: {noise.shape}")
    print(f"t=0:   x_t ≈ x_0? max diff = {(x_t[0] - x_0[0]).abs().max():.4f} (期望接近 0)")
    print(f"t=999: x_t 接近 noise? max diff = {(x_t[3] - noise[3]).abs().max():.4f} (期望小)")
    
    # 测试 training loss(用一个 fake model)
    class FakeModel(torch.nn.Module):
        def forward(self, x, t):
            return torch.zeros_like(x)
    
    fake_model = FakeModel()
    loss = diffusion.training_loss(fake_model, x_0)
    print(f"\nFake model loss: {loss.item():.4f} (期望接近 1.0,因为 MSE(noise, 0) ≈ 1)")
    
    print("\n✓ Diffusion scheduler 测试通过")
