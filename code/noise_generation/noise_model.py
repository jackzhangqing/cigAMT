"""
Conditional Noise-DiT for 1D AMT noise generation.

Input:
    x_t:          (B, 1, 2048)
    t:            (B,)
    type_id:      (B,)  noise class id
    length_ratio: (B,)  valid length / 2048
    source_id:    (B,)  optional, 0=real, 1=synthetic

Output:
    predicted diffusion noise, same shape as x_t.
"""

import torch
import torch.nn as nn

from model import DiTBlock, FinalLayer, TimestepEmbedder


class LengthEmbedder(nn.Module):
    """Embed scalar length_ratio into the DiT hidden dimension."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, length_ratio):
        if length_ratio.ndim == 1:
            length_ratio = length_ratio[:, None]
        return self.mlp(length_ratio.float())


class ConditionalNoiseDiT(nn.Module):
    """1D conditional DiT for single-channel noise clips."""

    def __init__(
        self,
        in_channels=1,
        input_length=2048,
        patch_size=8,
        hidden_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        num_noise_types=10,
        condition_on_source=False,
        num_sources=2,
        condition_on_length=True,
    ):
        super().__init__()
        assert input_length % patch_size == 0, (
            f"input_length({input_length}) must be divisible by patch_size({patch_size})"
        )

        self.in_channels = in_channels
        self.input_length = input_length
        self.patch_size = patch_size
        self.num_patches = input_length // patch_size
        self.hidden_dim = hidden_dim
        self.num_noise_types = num_noise_types
        self.condition_on_source = condition_on_source
        self.condition_on_length = condition_on_length

        self.patch_embed = nn.Conv1d(
            in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))

        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.type_embedder = nn.Embedding(num_noise_types, hidden_dim)
        self.length_embedder = LengthEmbedder(hidden_dim) if condition_on_length else None
        self.source_embedder = (
            nn.Embedding(num_sources, hidden_dim) if condition_on_source else None
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_dim, patch_size, in_channels)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.weight.view(self.patch_embed.weight.shape[0], -1))
        nn.init.zeros_(self.patch_embed.bias)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.type_embedder.weight, std=0.02)
        if self.source_embedder is not None:
            nn.init.normal_(self.source_embedder.weight, std=0.02)

        if self.length_embedder is not None:
            for module in self.length_embedder.mlp:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    nn.init.zeros_(module.bias)

        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, x):
        bsz, num_patches, _ = x.shape
        x = x.reshape(bsz, num_patches, self.in_channels, self.patch_size)
        x = x.permute(0, 2, 1, 3).reshape(bsz, self.in_channels, self.input_length)
        return x

    def forward(self, x, t, type_id, length_ratio, source_id=None):
        h = self.patch_embed(x).transpose(1, 2)
        h = h + self.pos_embed

        c = self.t_embedder(t) + self.type_embedder(type_id.long())
        if self.length_embedder is not None:
            c = c + self.length_embedder(length_ratio)
        if self.source_embedder is not None:
            if source_id is None:
                source_id = torch.zeros_like(type_id)
            c = c + self.source_embedder(source_id.long())

        for block in self.blocks:
            h = block(h, c)

        h = self.final_layer(h, c)
        return self.unpatchify(h)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    model = ConditionalNoiseDiT()
    x = torch.randn(4, 1, 2048)
    t = torch.randint(0, 1000, (4,))
    type_id = torch.tensor([0, 1, 2, 9])
    length_ratio = torch.tensor([100, 300, 1500, 2048]).float() / 2048
    with torch.no_grad():
        y = model(x, t, type_id=type_id, length_ratio=length_ratio)
    print("params(M):", model.num_parameters() / 1e6)
    print("in:", x.shape, "out:", y.shape)
