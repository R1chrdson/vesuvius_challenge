from typing import Tuple, Union

import torch
import torch.nn as nn


class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: Union[bool, str] = 'auto',
        ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        self.norm = nn.SyncBatchNorm(out_channels)

        if hasattr(self.norm, '_specify_ddp_gpu_num'):
            self.norm._specify_ddp_gpu_num(1)
            
        for param in self.norm.parameters():
            param.requires_grad = True

        self.activate = nn.ReLU()

    def forward(self, x: torch.Tensor, activate: bool = True, norm: bool = True) -> torch.Tensor:
        x = self.conv(x)
        if norm:
            x = self.norm(x)
        if activate:
            x = self.activate(x)

        return x
