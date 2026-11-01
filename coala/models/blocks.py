import gin
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import cached_conv as cc

from typing import Sequence


@gin.configurable
def normalization(module: nn.Module, mode: str = 'identity'):
    if mode == 'identity':
        return module
    elif mode == 'weight_norm':
        return weight_norm(module)
    else:
        raise Exception(f'Normalization mode {mode} not supported')


@gin.configurable
def get_activation(dim: int, name: str = 'silu'):
    if name.lower() == 'silu':
        return nn.SiLU()
    elif name.lower() == 'lrelu':
        return nn.LeakyReLU(.2)
    elif name.lower() == 'relu':
        return nn.ReLU()
    else:
        raise ValueError('unknown activation')


class Residual(nn.Module):

    def __init__(self, module, cumulative_delay=0):
        super().__init__()
        additional_delay = module.cumulative_delay
        self.aligned = cc.AlignBranches(
            module,
            nn.Identity(),
            delays=[additional_delay, 0],
        )
        self.cumulative_delay = additional_delay + cumulative_delay

    def forward(self, x):
        x_net, x_res = self.aligned(x)
        return x_net + x_res
    

class DilatedUnit(nn.Module):

    def __init__(self, dim: int, kernel_size: int, dilation: int):
        super().__init__()
        net = [
            get_activation(dim),
            normalization(
                cc.Conv1d(dim,
                          dim,
                          kernel_size=kernel_size,
                          dilation=dilation,
                          padding=cc.get_padding(
                              kernel_size,
                              dilation=dilation,
                          ))),
            nn.GroupNorm(1, dim),  # TODO: check if necessary
            get_activation(dim),
            # 1x1 conv for channel mixing before residual add
            normalization(cc.Conv1d(dim, dim, kernel_size=1, padding=cc.get_padding(1))),
        ]

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = net[1].cumulative_delay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

class TCN(nn.Module):

    def __init__(self, latent_dim: int, hidden_dim: int, n_out: int, kernel_size: int, dilations: Sequence[int], proj_dim=None):
        super().__init__()

        num_channels = hidden_dim
        self.proj_dim = latent_dim if proj_dim is None else proj_dim

        net = [
            normalization(
                cc.Conv1d(
                    latent_dim,
                    num_channels,
                    kernel_size=1, # kernel_size * 2 + 1,
                    padding=cc.get_padding(1),  # kernel_size * 2 + 1),
                )),
        ]

        for d_list in dilations:
            for d in d_list:
                net.append(
                    Residual(
                        DilatedUnit(
                            dim=num_channels,
                            kernel_size=kernel_size,
                            dilation=d,
                        )))

        net.append(get_activation(num_channels))
        net.append(
            normalization(
                cc.Conv1d(
                    num_channels,
                    self.proj_dim * n_out,
                    kernel_size=kernel_size,
                    padding=cc.get_padding(kernel_size),
                    bias=True,
                )))

        self.net = cc.CachedSequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x
    

class CausalCNN(nn.Module):

    def __init__(self, latent_dim: int, hidden_dim: int, n_out: int, kernel_size: int, proj_dim=None):
        super().__init__()

        num_channels = hidden_dim
        self.proj_dim = latent_dim if proj_dim is None else proj_dim

        net = [
            cc.Conv1d(
                latent_dim,
                num_channels,
                kernel_size=kernel_size, # 1, 
                padding=cc.get_padding(kernel_size), 
            ),
        ]
        if kernel_size > 1:
            net.append(get_activation(num_channels))
            net.append(
                cc.Conv1d(
                    num_channels,
                    num_channels,
                    kernel_size=kernel_size,
                    padding=cc.get_padding(kernel_size),
                ))
        net.append(get_activation(num_channels))
        net.append(
            cc.Conv1d(
                num_channels,
                self.proj_dim * n_out,
                kernel_size=1,
                padding=cc.get_padding(1),
            ))

        self.net = cc.CachedSequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x


class ChannelMixingUnit(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        net = [
            normalization(cc.Conv1d(dim, dim * 2, kernel_size=1)),
            get_activation(dim * 2),
            normalization(cc.Conv1d(dim * 2, dim, kernel_size=1)),
        ]

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CausalCNN2(nn.Module):

    def __init__(self, latent_dim: int, hidden_dim: int, proj_dim=None):
        super().__init__()

        self.proj_dim = latent_dim if proj_dim is None else proj_dim

        net = [
            cc.Conv1d(
                latent_dim,
                hidden_dim,
                kernel_size=3, 
                padding=cc.get_padding(3), 
            ),
            # get_activation(hidden_dim),
        ]

        net.append(
            Residual(
                ChannelMixingUnit(dim=hidden_dim)
            ))
            
        net.append(
            cc.Conv1d(
                hidden_dim,
                self.proj_dim,
                kernel_size=1,
                padding=cc.get_padding(1),
            ))

        self.net = cc.CachedSequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x
