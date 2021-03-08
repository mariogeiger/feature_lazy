# pylint: disable=E1101, C, arguments-differ
"""
Defines three architectures:
- Fully connecetd `FC`
- Convolutional `CV`
- And a resnet `Wide_ResNet`
"""
import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FC(nn.Module):
    def __init__(self, d, h, c, L, act, bias=False, last_bias=False, var_bias=0):
        super().__init__()

        hh = d
        for i in range(L):
            W = torch.randn(h, hh)

            # next two line are here to avoid memory issue when computing the kernel
            n = max(1, 128 * 256 // hh)
            W = nn.ParameterList([nn.Parameter(W[j: j + n]) for j in range(0, len(W), n)])

            setattr(self, "W{}".format(i), W)
            if bias:
                self.register_parameter("B{}".format(i), nn.Parameter(torch.randn(h).mul(var_bias**0.5)))
            hh = h

        self.register_parameter("W{}".format(L), nn.Parameter(torch.randn(c, hh)))
        if last_bias:
            self.register_parameter("B{}".format(L), nn.Parameter(torch.randn(c).mul(var_bias**0.5)))

        self.L = L
        self.act = act
        self.bias = bias
        self.last_bias = last_bias

    def forward(self, x):
        for i in range(self.L + 1):
            W = getattr(self, "W{}".format(i))

            if isinstance(W, nn.ParameterList):
                W = torch.cat(list(W))

            if self.bias and i < self.L:
                B = self.bias * getattr(self, "B{}".format(i))
            elif self.last_bias and i == self.L:
                B = self.last_bias * getattr(self, "B{}".format(i))
            else:
                B = 0

            h = x.size(1)

            if i < self.L:
                x = x @ (W.t() / h ** 0.5)
                x = self.act(x + B)
            else:
                x = x @ (W.t() / h) + B

        if x.shape[1] == 1:
            return x.view(-1)
        return x


class FixedWeights(nn.Module):
    def __init__(self, d, h, act, bias):
        super().__init__()

        self.register_buffer("W0", torch.randn(h, d))
        self.B = nn.Parameter(torch.zeros(h))
        self.W = nn.Parameter(torch.randn(h))

        self.act = act
        self.bias = bias

    def forward(self, x):
        d = x.size(1)
        B = self.bias * self.B
        h = len(B)
        return self.act(x @ (self.W0.T / d**0.5) + B) @ (self.W / h)


class FixedAngles(nn.Module):
    def __init__(self, d, h, act, bias):
        super().__init__()

        W0 = torch.randn(h, d)
        r = W0.norm(dim=1, keepdim=True)
        self.r = nn.Parameter(r)
        self.register_buffer("W0", W0 / r)
        self.B = nn.Parameter(torch.zeros(h))
        self.W = nn.Parameter(torch.randn(h))

        self.act = act
        self.bias = bias

    def forward(self, x):
        d = x.size(1)
        B = self.bias * self.B
        h = len(B)
        return self.act(x @ ((self.r * self.W0).T / d**0.5) + B) @ (self.W / h)


class Conv1d(nn.Module):
    def __init__(self, d, h, act, bias):
        super().__init__()

        self.W = nn.Parameter(torch.randn(h, 1, d))
        self.B = nn.Parameter(torch.randn(h))
        self.C = nn.Parameter(torch.randn(h))

        self.act = act
        self.bias = bias

    def forward(self, x):
        d = x.size(1)
        B = self.bias * self.B
        h = len(B)
        x = torch.cat((x, x[:, :-1]), dim=1)
        x = x.reshape(x.size(0), 1, d + d - 1)
        return (self.act(F.conv1d(x, self.W / d**0.5, B)).sum(dim=2) / d) @ (self.C / h)


class CV(nn.Module):
    def __init__(self, d, h, L1, L2, act, h_base, fsz, pad, stride_first):
        super().__init__()

        h1 = d
        for i in range(L1):
            h2 = round(h * h_base ** i)
            for j in range(L2):
                W = nn.ParameterList([nn.Parameter(torch.randn(h1, fsz, fsz)) for _ in range(h2)])
                setattr(self, "W{}_{}".format(i, j), W)
                h1 = h2

        self.W = nn.Parameter(torch.randn(h1))

        self.L1 = L1
        self.L2 = L2
        self.act = act
        self.pad = pad
        self.stride_first = stride_first

    def forward(self, x):
        for i in range(self.L1):
            for j in range(self.L2):
                assert x.size(2) >= 5 and x.size(3) >= 5
                W = getattr(self, "W{}_{}".format(i, j))
                W = torch.stack(list(W))

                stride = 2 if j == 0 and (i > 0 or self.stride_first) else 1
                h = W[0].numel()
                x = nn.functional.conv2d(x, W / h ** 0.5, None, stride=stride, padding=self.pad)
                x = self.act(x)

        x = x.flatten(2).mean(2)

        W = self.W
        h = len(W)
        x = x @ (W / h)
        return x.view(-1)


class conv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        w = torch.randn(out_planes, in_planes, kernel_size, kernel_size)
        n = max(1, 256**2 // w[0].numel())
        self.w = nn.ParameterList([nn.Parameter(w[j: j + n]) for j in range(0, len(w), n)])

        self.b = nn.Parameter(torch.zeros(out_planes)) if bias else None

        self.stride = stride
        self.padding = padding

    def forward(self, x):
        w = torch.cat(list(self.w))
        h = w[0].numel()
        return F.conv2d(x, w / h ** 0.5, self.b, self.stride, self.padding)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, act, stride=1, mix_angle=45):
        super().__init__()
        self.conv1 = conv(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.conv2 = conv(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = conv(in_planes, planes, kernel_size=1, stride=stride, bias=True)

        self.act = act
        self.mix_angle = mix_angle

    def forward(self, x):
        out = self.conv1(self.act(x))
        out = self.conv2(self.act(out))
        cut = self.shortcut(x)

        a = self.mix_angle * math.pi / 180
        out = math.cos(a) * cut + math.sin(a) * out

        return out


class Wide_ResNet(nn.Module):
    def __init__(self, d, depth, h, act, num_classes, mix_angle=45):
        super().__init__()

        assert (depth % 6 == 4), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) // 6

        nStages = [16, 16 * h, 32 * h, 64 * h]
        block = functools.partial(wide_basic, act=act, mix_angle=mix_angle)

        self.conv1 = conv(d, nStages[0], kernel_size=3, stride=1, padding=1, bias=True)
        self.in_planes = nStages[0]

        self.layer1 = self._wide_layer(block, nStages[1], n, stride=1)
        self.layer2 = self._wide_layer(block, nStages[2], n, stride=2)
        self.layer3 = self._wide_layer(block, nStages[3], n, stride=2)
        self.linear = nn.Parameter(torch.randn(num_classes, nStages[3]))
        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.act = act

    def _wide_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride=stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.act(out)
        out = out.flatten(2).mean(2)

        h = self.linear.size(1)
        out = F.linear(out, self.linear / h, self.bias)

        if out.size(1) == 1:
            out = out.flatten(0)

        return out
