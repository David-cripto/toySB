import torch as th
from torch import nn
import math

class FourierFeatures(nn.Module):
    def __init__(self, in_features: int, out_features: int, std: float = 1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(th.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return th.cat([f.cos(), f.sin()], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, prob: float = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(prob),
            nn.ReLU(),
        )
        self.residual = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.model(x) + self.residual(x)

class SBModel(nn.Module):
    def __init__(self, dim: int, scale: int = 32, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.timestep_embed = FourierFeatures(1, dim, std=0.2)

        self.layers = nn.Sequential(
            ResBlock(2*dim, dim * scale),
            ResBlock(dim * scale, dim * scale),
            ResBlock(dim * scale, dim * scale * 2),
            ResBlock(dim * scale * 2, dim * scale * 2),
            ResBlock(dim * scale * 2, dim * scale * 4),
            ResBlock(dim * scale * 4, dim * scale * 4),
            ResBlock(dim * scale * 4, dim * scale * 2),
            ResBlock(dim * scale * 2, dim * scale * 2),
            ResBlock(dim * scale * 2, dim * scale),
            ResBlock(dim * scale, dim * scale),
            ResBlock(dim * scale, dim),
        )
    
    def forward(self, x, time):
        timestep_embed = self.timestep_embed(time[: None])
        return self.layers(th.cat([x, timestep_embed], dim=1))