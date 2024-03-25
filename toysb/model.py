import torch as th
from torch import nn


class SB(nn.Module):
    def __init__(self, dim: int, scale: int = 32, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
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

        self.layers = nn.Sequential(
            ResBlock(dim, dim * scale),
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
    
    def forward(self, x):
        return self.layers(x)