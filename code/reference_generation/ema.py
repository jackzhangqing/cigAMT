"""
Exponential Moving Average for model weights
=============================================
扩散模型的标准做法:维护一个 EMA 副本,采样时用 EMA 模型(质量明显更好)

EMA 模型不参与梯度训练，只在每次 optimizer.step() 后用“指数滑动平均”的方式缓慢跟随主模型

扩散模型采样时通常使用 EMA 模型，因为它比主模型当前权重更平滑、更稳定，生成质量通常更好。
"""

from copy import deepcopy
import torch


class EMA:
    """模型权重的 EMA shadow"""
    
    def __init__(self, model, decay=0.9999):
        """
        model: 主训练模型
        decay: EMA 衰减率(0.999-0.9999 之间,越大越慢)
        """
        self.decay = decay
        self.shadow_model = deepcopy(model)
        self.shadow_model.eval()
        for p in self.shadow_model.parameters():
            p.requires_grad_(False)
    
    @torch.no_grad()
    def update(self, model):
        """每次 optimizer.step() 后调用"""
        for s_p, m_p in zip(self.shadow_model.parameters(), model.parameters()):
            s_p.data.mul_(self.decay).add_(m_p.data, alpha=1 - self.decay)
        # buffer(LayerNorm 的 running stats 等)直接复制
        for s_b, m_b in zip(self.shadow_model.buffers(), model.buffers()):
            s_b.data.copy_(m_b.data)
    
    def state_dict(self):
        return self.shadow_model.state_dict()
    
    def load_state_dict(self, state_dict):
        self.shadow_model.load_state_dict(state_dict)
