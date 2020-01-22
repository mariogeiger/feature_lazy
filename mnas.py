# pylint: disable=arguments-differ, no-member, missing-docstring, invalid-name, line-too-long
"""
Defines three architectures:
- Fully connecetd `FC`
- Convolutional `CV`
- And a resnet `Wide_ResNet`
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from swish import SwishJit


class NTKLinear(nn.Module):
    def __init__(self, in_chs, out_chs, bias=True):
        super().__init__()

        w = torch.randn(out_chs, in_chs)
        n = max(1, 256**2 // w[0].numel())
        self.w = nn.ParameterList([nn.Parameter(w[j: j + n]) for j in range(0, len(w), n)])
        self.b = nn.Parameter(torch.zeros(out_chs)) if bias else None

    def forward(self, x):
        w = torch.cat(list(self.w))
        h = w[0].numel()
        return F.linear(x, w / h ** 0.5, self.b)


class NTKConv(nn.Module):
    def __init__(self, in_chs, out_chs, k, s=1, p=0, g=1, bias=True):
        super().__init__()

        w = torch.randn(out_chs, in_chs // g, k, k)
        n = max(1, 256**2 // w[0].numel())
        self.w = nn.ParameterList([nn.Parameter(w[j: j + n]) for j in range(0, len(w), n)])

        self.b = nn.Parameter(torch.zeros(out_chs)) if bias else None

        self.s = s
        self.p = p
        self.g = g

    def forward(self, x):
        w = torch.cat(list(self.w))
        h = w[0].numel()
        return F.conv2d(x, w / h ** 0.5, self.b, self.s, self.p, 1, self.g)


class DepthwiseSeparableConv(nn.Module):
    """ DepthwiseSeparable block """
    def __init__(self, in_chs, out_chs, k=3, s=1, p=1, act=SwishJit):
        super().__init__()
        self.conv_dw = NTKConv(in_chs, in_chs, k, s, p, g=in_chs, bias=False)
        self.act1 = act()
        self.conv_pw = NTKConv(in_chs, out_chs, k=1, bias=False)
        self.act2 = act()

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.act1(x)

        x = self.conv_pw(x)
        x = self.act2(x)
        return x


class InvertedResidual(nn.Module):
    """ Inverted residual block """
    def __init__(self, in_chs, out_chs, k=3, s=1, p=1, act=SwishJit, noskip=False, exp_ratio=6.0):
        super().__init__()
        mid_chs = round(in_chs * exp_ratio)
        self.has_residual = (in_chs == out_chs and s == 1) and not noskip

        # Point-wise expansion
        self.conv_pw = NTKConv(in_chs, mid_chs, 1, bias=False)
        self.act1 = act()

        # Depth-wise convolution
        self.conv_dw = NTKConv(mid_chs, mid_chs, k, s, p, g=mid_chs, bias=False)
        self.act2 = act()

        # Point-wise linear projection
        self.conv_pwl = NTKConv(mid_chs, out_chs, 1, bias=False)

    def forward(self, x):
        residual = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.act2(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)

        if self.has_residual:
            x = (x + residual) / 2 ** 0.5
        return x


class Mem:
    def __init__(self, x=None):
        self.x = x
    def __call__(self, x=None):
        self.x = x or self.x
        return self.x


class MnasNetLike(nn.Module):
    def __init__(self, d, h, n_blocks=2, n_layers=2):
        super().__init__()
        c = Mem()

        self.conv_stem = NTKConv(c(d), c(round(4 * h)), k=5, s=2, p=2, bias=False)  # 16x16
        self.act1 = SwishJit()

        self.blocks = nn.Sequential(
            DepthwiseSeparableConv(c(), c(round(2 * h)), k=5, s=1, p=2),
        )
        for i in range(n_blocks):
            self.blocks.add_module(InvertedResidual(c(), c(round((2 * i + 1) * h)), k=5, s=2, p=2, exp_ratio=3.0))
            for j in range(n_layers - 1):
                self.blocks.add_module(f"ir{i}_{j}", InvertedResidual(c(), c(), k=5, s=1, p=2, exp_ratio=3.0))

        self.conv_head = NTKConv(c(), c(round(20 * h)), k=1, s=1, bias=False)
        self.act2 = SwishJit()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = NTKLinear(c(), 1, bias=True)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x.view(-1)
