from torch import nn
from I2SB.guided_diffusion.script_util import create_model
import torch as th
from toysb.utils import expand_to_planes
import math

class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)


class ResConvBlock(ResidualBlock):
    def __init__(self, c_in, c_mid, c_out, dropout_last=True):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv2d(c_in, c_mid, kernel_size=3, padding=1),
            nn.Dropout2d(0.1, inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_out, kernel_size=3, padding=1),
            nn.Dropout2d(0.1, inplace=True) if dropout_last else nn.Identity(),
            nn.ReLU(inplace=True),
        ], skip)


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return th.cat([self.main(input), self.skip(input)], dim=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(th.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return th.cat([f.cos(), f.sin()], dim=-1)


class Unet(nn.Module):
    def __init__(self, c_in: int, c_inter: int = 64, time_dim: int = 16):
        super().__init__()

        # The inputs to timestep_embed will approximately fall into the range
        # -10 to 10, so use std 0.2 for the Fourier Features.
        self.timestep_embed = FourierFeatures(1, time_dim, std=0.2)

        self.net = nn.Sequential(   
            ResConvBlock(c_in + time_dim, c_inter, c_inter),
            ResConvBlock(c_inter, c_inter, c_inter),
            SkipBlock([
                nn.AvgPool2d(2),  
                ResConvBlock(c_inter, c_inter * 2, c_inter * 2),
                ResConvBlock(c_inter * 2, c_inter * 2, c_inter * 2),
                SkipBlock([
                    nn.AvgPool2d(2),
                    ResConvBlock(c_inter * 2, c_inter * 4, c_inter * 4),
                    ResConvBlock(c_inter * 4, c_inter * 4, c_inter * 4),
                    ResConvBlock(c_inter * 4, c_inter * 4, c_inter * 2),
                    nn.Upsample(scale_factor=2),
                ]),
                ResConvBlock(c_inter * 4, c_inter * 2, c_inter * 2),
                ResConvBlock(c_inter * 2, c_inter * 2, c_inter),
                nn.Upsample(scale_factor=2),
            ]),
            ResConvBlock(c_inter * 2, c_inter, c_inter),
            ResConvBlock(c_inter, c_inter, 3, dropout_last=False),
        )

    def forward(self, input, log_snrs):
        timestep_embed = expand_to_planes(self.timestep_embed(log_snrs[:, None]), input.shape)
        return self.net(th.cat([input, timestep_embed], dim=1))

def get_model(*args, **kwargs):
    return create_model(*args, **kwargs)