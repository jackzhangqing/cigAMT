"""
GeoAMT-DiT: 时域 1D Diffusion Transformer

扩散模型训练时通常会让模型学习：模型预测噪声 ≈ 实际加入的噪声

==========================================
无条件生成 4 通道 AMT 信号 (4, 2048)

架构(参考 SWaG / TSDiT,但更轻量):
    Input (B, 4, 2048)
      ↓ Patch Embed (Conv1D, k=8 s=8)   → (B, 256, hidden_dim)
      ↓ + 位置编码
      ↓
    DiT Block × depth(每层包含):
      - AdaLN(只用 timestep 调制 scale/shift/gate)
      - Multi-Head Self-Attention
      - AdaLN + MLP
      ↓
    Final Layer Norm + Linear
      ↓
    Un-patchify → Output (B, 4, 2048)

无条件版本:只用 timestep 做 AdaLN 调制
后续可加入 source/model_type 等条件输入

这段代码实现了一个用于扩散模型的 1D DiT 网络。
它把 (B,4,2048) 的 AMT 时间序列切成 256 个 patch token，通过 12 层带 timestep AdaLN 调制的 Transformer 学习噪声预测，
最后再把 token 还原成 (B,4,2048)。
训练好后，它可以作为扩散模型的噪声预测器，用于从随机噪声逐步生成新的四通道 AMT 信号。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Timestep 编码(标准 sinusoidal + MLP)
# 扩散模型里，每一步的噪声强度不同，所以模型必须知道当前是第几个扩散时间步。
# ============================================================
class TimestepEmbedder(nn.Module):
    """把扩散时间步 t 编码为 hidden_dim 维向量"""
    
    def __init__(self, hidden_dim, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(    # MLP 映射到 hidden_dim
            nn.Linear(frequency_embedding_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):  
        """
        t: (B,) tensor of timesteps (integers or float)
        return: (B, dim) sinusoidal embeddings
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * 
            torch.arange(0, half, device=t.device).float() / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        # 把整数时间步 t 编码成正弦/余弦向量。
        # 用不同频率的 sin 和 cos 表示时间步。
        return embedding
    
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)

        # MLP 映射到 hidden_dim
        return self.mlp(t_freq)


# ============================================================
# DiT Block(带 AdaLN-zero,只用 t 调制)
# ============================================================
def modulate(x, shift, scale):
    """AdaLN: x * (1 + scale) + shift"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    普通 Transformer Block：
    LayerNorm + Attention + MLP

    DiT Block：它专门为扩散模型服务。因为扩散模型每一步 t 的噪声程度不同，所以需要用 timestep embedding 去调制网络。
    LayerNorm + AdaLN(timestep调制) + Attention + MLP


    每个DiTBlock都在做:
        根据当前扩散时间步 t，
        让 256 个 patch token 通过 Attention 交换信息，
        再通过 MLP 提取非线性特征，
        最后输出更适合预测噪声的 token 表示。


    标准 DiT block:
        - LayerNorm + AdaLN scale/shift
        - Multi-Head Self-Attention
        - AdaLN gate(乘到 attn 输出)
        - LayerNorm + AdaLN scale/shift  
        - MLP
        - AdaLN gate


    输入：256 个时间片 token
    条件：当前扩散时间步 t
    第一道工序：Attention
    让所有时间片互相交流，判断哪些片段相关。
    第二道工序：MLP
    对每个时间片的特征进一步加工。
    控制系统：AdaLN
    根据当前噪声强度 t，调整 Attention 和 MLP 的处理方式。
    输出：更新后的 256 个 token。

    DiTBlock 是 GeoAMT-DiT 的核心 Transformer 层。
    它接收 (B,256,384) 的 patch token 和 (B,384) 的 timestep 条件向量，
    通过 AdaLN 根据扩散时间步动态调制 Attention 和 MLP 两个分支，使模型在不同噪声阶段都能学习如何预测噪声。


    每个 DiTBlock 输入输出形状都不变：(B, 256, 384) -> (B, 256, 384)

    所以 DiTBlock 的作用就是：不断处理这 256 个时间片 token，让它们之间相互交流，同时根据扩散时间步 t 调整当前层的处理方式。

    # 一个DiTBlock内部有两部分：  Attention 负责让不同时间 patch 之间相互关联；MLP 负责对每个 token 的特征做非线性变换。
    # Attention 分支
    x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
    attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
    x = x + gate_msa.unsqueeze(1) * attn_out

    # MLP 分支
    x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
    x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
    """
    
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )  # num_heads    每个 head 会单独做一次注意力计算，然后再拼接回 384 维。
        # 默认6个头
        # head 1：关注局部相邻时间片
        # head 2：关注长距离周期关系
        # head 3：关注异常尖峰或噪声段
        # head 4：关注 Ex/Ey 相关特征
        # head 5：关注 Hx/Hy 相关特征
        # head 6：关注整体波形趋势
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden = int(hidden_dim * mlp_ratio)
        '''
        优点：模型表达能力更强
        缺点：参数量更大，显存更多，训练更慢，也更容易过拟合
        
        为什么要先放大再缩回去？
            MLP 分支的作用是对每个 token 的特征做非线性变换。
            先把 384 维放大到 1536 维，可以让模型有更强的表达能力；经过 GELU 激活后，再压回 384 维，方便和原来的 x 做残差连接。
        '''
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_hidden, hidden_dim),
        )
        
        # AdaLN 调制层:从 t 向量生成 6 组参数(shift/scale/gate × attn/mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True),
        )
    
    def forward(self, x, c):
        """
        x: (B, N, D) tokens
        c: (B, D)    condition vector (timestep embed)

        普通 Transformer Block 一般是：
            LayerNorm → Attention → 残差
            LayerNorm → MLP → 残差

        这里的 DiTBlock 是 AdaLN-zero 版本:  自适应 LayerNorm 调制
        LayerNorm → AdaLN调制 → Attention → gate → 残差
        LayerNorm → AdaLN调制 → MLP → gate → 残差
        """
        # c控制条件告诉模型：当前是扩散过程中的第几步  当前噪声强度大概是多少  模型应该怎样去预测噪声
        # 这里的c就是扩散时间步t  t 很大：信号很噪，模型主要学粗略结构    t 很小：信号较干净，模型主要学细节修正
        # 生成 6 组调制参数:
        # shift_msa : Attention 分支的平移参数
        # scale_msa : Attention 分支的缩放参数
        # gate_msa  : Attention 分支的门控参数

        # shift_mlp : MLP 分支的平移参数
        # scale_mlp : MLP 分支的缩放参数
        # gate_mlp  : MLP 分支的门控参数
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)  # 普通 LayerNorm 只是归一化，而 AdaLN 是在归一化之后，根据条件 c 再调整特征。
        
        '''
        def modulate(x, shift, scale):
            """AdaLN: x * (1 + scale) + shift"""
            return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        '''
        # Attention 分支
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)  # x: (B, N, D) tokens

        # Q = x_norm
        # K = x_norm
        # V = x_norm
        # 这就叫 Self-Attention。让同一个序列内部的 token 互相看、互相建立关系。
        # Q、K、V 都来自 x_norm，但经过不同线性层变换后，数值一般不一样。

        # attn(query=x1, key=x2, value=x2) : 交叉注意力 Cross-Attention, Q 来自一个序列，K、V 来自另一个序列：一个序列去关注另一个序列的信息
        '''
        Attention(Q, K, V) = softmax(QKᵀ / √d) V
            Q：Query，查询向量
            K：Key，键向量
            V：Value，值向量
            d：每个 head 的维度
            3. Q、K、V 分别在干什么？

            可以这样理解：
            Q：我想找什么信息
            某个 token 会生成自己的 Query。
            比如第 10 个 token：Q_10：第10个时间片想去找哪些相关信息？

            K：我有什么特征可以被别人匹配
            每个 token 都会生成自己的 Key。
            比如第 50 个 token：K_50：第50个时间片有什么特征？

            V：如果别人关注我，我提供什么内容
            每个 token 也会生成自己的 Value。
            比如第 50 个 token：V_50：如果其他 token 关注我，我把什么信息传给它？

        第10个 token 用自己的 Q 去匹配所有 token 的 K；
        匹配程度越高，就越关注那个 token 的 V；
        最后把所有被关注 token 的 V 加权求和，得到第10个 token 的新表示。

        x_norm.shape = (B, 256, 384)
        B      = batch size
        256    = token 数量
        384    = 每个 token 的特征维度
        设定num_heads = 6，所以每个 head 的维度是：384 / 6 = 64


        Q 输入 = x_norm
        K 输入 = x_norm
        V 输入 = x_norm
        nn.MultiheadAttention 内部会自动做线性变换：
        Q = x_norm @ W_Q + b_Q
        K = x_norm @ W_K + b_K
        V = x_norm @ W_V + b_V
        '''
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)  # 对 256 个 token 做多头自注意力。
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # MLP 分支
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        
        return x


# ============================================================
# Final Layer:还原回波形  把 token 输出成 patch 数据
# ============================================================
class FinalLayer(nn.Module):
    """LayerNorm + AdaLN + Linear,输出 patch_size × out_channels 维"""
    
    def __init__(self, hidden_dim, patch_size, out_channels):
        super().__init__()

        # 普通 LayerNorm 默认有可学习参数：gamma 和 beta   也就是缩放和平移。
        # 但是这里不用普通的固定 gamma/beta，而是用 timestep 条件生成的：
        # LayerNorm 只做归一化；
        # 缩放和平移交给 AdaLN 根据 timestep 动态生成。
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)

        self.linear = nn.Linear(hidden_dim, patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True),
        )
    
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)   # (B, N, patch_size * out_channels)


# ============================================================
# 主模型:GeoAMT-DiT
# ============================================================
class GeoAMT_DiT(nn.Module):
    """
    时域 1D DiT,用于无条件 4 通道 AMT 信号生成
    
    输入:  x of shape (B, 4, L)         L=2048
    输出:  ε predicted, same shape
    """
    
    def __init__(self,
                 in_channels=4,
                 input_length=2048,
                 patch_size=8,
                 hidden_dim=384,
                 depth=12,
                 num_heads=6,
                 mlp_ratio=4.0):
        super().__init__()
        
        assert input_length % patch_size == 0, \
            f"input_length({input_length}) 必须能被 patch_size({patch_size}) 整除"
        
        self.in_channels = in_channels
        self.input_length = input_length
        self.patch_size = patch_size
        self.num_patches = input_length // patch_size
        self.hidden_dim = hidden_dim
        
        # Patch embedding:(B, C, L) -> (B, num_patches, hidden_dim)
        self.patch_embed = nn.Conv1d(
            in_channels, hidden_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # 可学习的位置编码
        # # 因为 Transformer 本身不像 CNN/RNN 那样天然知道顺序，如果没有位置编码，它只知道有 256 个 token，但不知道谁在前谁在后。
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_dim)
        )
        
        # Timestep embedding
        self.t_embedder = TimestepEmbedder(hidden_dim) # 扩散模型里，每一步的噪声强度不同，所以模型必须知道当前是第几个扩散时间步。
        
        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        # Final layer
        self.final_layer = FinalLayer(hidden_dim, patch_size, in_channels)
        
        # 初始化
        self.initialize_weights()
    
    def initialize_weights(self):
        """标准 DiT 初始化"""
        # 位置编码:Xavier
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Patch embed:类似 ViT
        w = self.patch_embed.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.patch_embed.bias)
        
        # Timestep embedder MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        # AdaLN-zero:所有 adaLN_modulation 最后一层置零
        # 这让模型在初始时表现为 identity,训练初期更稳定
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        
        # Final linear:零初始化(输出初始为 0)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)
    
    def unpatchify(self, x):
        """
        把 (B, N, patch_size * C) 还原成 (B, C, L)
        """
        B, N, _ = x.shape
        # (B, N, patch_size * C) -> (B, N, C, patch_size)
        x = x.reshape(B, N, self.in_channels, self.patch_size)
        # -> (B, C, N, patch_size) -> (B, C, N * patch_size = L)
        x = x.permute(0, 2, 1, 3).reshape(B, self.in_channels, self.input_length)
        return x
    
    def forward(self, x, t):
        """
        x: (B, 4, 2048) 含噪信号
        t: (B,) 时间步
        return: (B, 4, 2048) 预测的噪声
        """
        # Patch embed
        # (B, 4, 2048) -> (B, hidden_dim, 256) -> (B, 256, hidden_dim)  # 256 个 token  每个 token 是 384 维特征
        h = self.patch_embed(x).transpose(1, 2)  

        # 因为 Transformer 本身不像 CNN/RNN 那样天然知道顺序，如果没有位置编码，它只知道有 256 个 token，但不知道谁在前谁在后。
        h = h + self.pos_embed   # 位置编码：告诉模型token的顺序信息
        
        # 条件向量(只有 timestep)
        c = self.t_embedder(t)
        
        # 走 DiT blocks
        for block in self.blocks:
            h = block(h, c)
        
        # Final layer + unpatchify
        h = self.final_layer(h, c)        # (B, N, patch_size * C)  FinalLayer：把 token 输出成 patch 数据

        out = self.unpatchify(h)          # (B, C, L)   unpatchify：还原回原始波形长度
        
        return out
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 快速测试
# ============================================================
if __name__ == '__main__':
    # 实例化
    model = GeoAMT_DiT(
        in_channels=4,
        input_length=2048,
        patch_size=8,
        hidden_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
    )
    
    print(f"模型参数量: {model.num_parameters() / 1e6:.2f} M")
    
    # 测试前向
    B = 4
    x = torch.randn(B, 4, 2048)
    t = torch.randint(0, 1000, (B,))
    
    with torch.no_grad():
        out = model(x, t)
    
    print(f"输入:  {x.shape}")
    print(f"输出:  {out.shape}")
    print(f"输出范围: [{out.min().item():.4f}, {out.max().item():.4f}]")
    assert out.shape == x.shape, "输出 shape 不匹配!"
    print("✓ 前向通过")

'''
模型参数量: 32.59 M
输入:  torch.Size([4, 4, 2048])
输出:  torch.Size([4, 4, 2048])

输出范围: [0.0000, 0.0000]:
    扩散模型训练目标通常是：
    loss = MSE(预测噪声, 真实噪声)  在不同噪声强度 t 下，从含噪 AMT 信号 x_t 中预测噪声 ε
    一开始模型什么都不会，输出 0 是比较稳定的初始状态。
    训练开始后，loss 会推动 final layer 和前面的 DiTBlock 慢慢学会预测噪声。
    所以初始输出全 0 不影响训练，反而有助于训练早期稳定。

✓ 前向通过
'''
