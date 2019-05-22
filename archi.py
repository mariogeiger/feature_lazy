# pylint: disable=E1101, C, arguments-differ
import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FC(nn.Module):
    def __init__(self, d, h, L, act, beta):
        super().__init__()

        f = d
        for i in range(L):
            W = torch.randn(h, f)

            # next two line are here to avoid memory issue when computing the kernel
            n = max(1, 256**2 // f)
            W = nn.ParameterList([nn.Parameter(W[j: j+n]) for j in range(0, len(W), n)])

            setattr(self, "W{}".format(i), W)

            self.register_parameter("b{}".format(i), nn.Parameter(torch.zeros(h)))
            f = h

        self.register_parameter("W{}".format(L), nn.Parameter(torch.randn(1, f)))
        self.register_parameter("b{}".format(L), nn.Parameter(torch.zeros(1)))

        self.L = L
        self.act = act
        self.beta = beta

    def forward(self, x):
        for i in range(self.L + 1):
            W = getattr(self, "W{}".format(i))
            b = getattr(self, "b{}".format(i))

            W = torch.cat(list(W))

            f = x.size(1)
            x = x @ (W.t() / f ** 0.5) + self.beta * b

            if i < self.L:
                x = self.act(x)

        return x.view(-1)


class CV(nn.Module):
    def __init__(self, d, h, L1, L2, act, h_base, fsz, beta, pad, stride_first):
        super().__init__()

        f1 = d
        for i in range(L1):
            f2 = int(h * h_base ** i)
            for j in range(L2):
                W = nn.ParameterList([nn.Parameter(torch.randn(f1, fsz, fsz)) for _ in range(f2)])
                setattr(self, "W{}_{}".format(i, j), W)

                self.register_parameter("b{}_{}".format(i, j), nn.Parameter(torch.zeros(f2)))
                f1 = f2

        self.register_parameter("W", nn.Parameter(torch.randn(f1)))
        self.register_parameter("b", nn.Parameter(torch.zeros(())))

        self.L1 = L1
        self.L2 = L2
        self.act = act
        self.beta = beta
        self.pad = pad
        self.stride_first = stride_first

    def forward(self, x):
        for i in range(self.L1):
            assert x.size(2) >= 5 and x.size(3) >= 5
            for j in range(self.L2):
                W = getattr(self, "W{}_{}".format(i, j))
                b = getattr(self, "b{}_{}".format(i, j))

                W = torch.stack(list(W))

                stride = 2 if j == 0 and (i > 0 or self.stride_first) else 1
                f = W[0].numel()
                x = nn.functional.conv2d(x, W / f ** 0.5, self.beta * b, stride=stride, padding=self.pad)
                x = self.act(x)

        x = x.flatten(2).mean(2)

        W = getattr(self, "W")
        b = getattr(self, "b")
        f = len(W)
        x = x @ (W / f ** 0.5) + self.beta * b
        return x.view(-1)


class conv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, bias=True, gain=1):
        super().__init__()

        w = torch.randn(out_planes, in_planes, kernel_size, kernel_size)
        n = max(1, 256**2 // w[0].numel())
        self.w = nn.ParameterList([nn.Parameter(w[j: j + n]) for j in range(0, len(w), n)])

        self.b = nn.Parameter(torch.zeros(out_planes)) if bias else None

        self.stride = stride
        self.padding = padding
        self.gain = gain

    def forward(self, x):
        w = torch.cat(list(self.w))
        f = self.gain / w[0].numel() ** 0.5
        return F.conv2d(x, f * w, self.b, self.stride, self.padding)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, stride=1, mix_angle=45):
        super().__init__()
        self.conv1 = conv(in_planes, planes, kernel_size=3, padding=1, bias=True, gain=2 ** 0.5)
        self.conv2 = conv(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True, gain=2 ** 0.5)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = conv(in_planes, planes, kernel_size=1, stride=stride, bias=True, gain=1)

        self.mix_angle = mix_angle

    def forward(self, x):
        out = self.conv1(F.relu(x))
        out = self.conv2(F.relu(out))
        cut = self.shortcut(x)

        a = self.mix_angle * math.pi / 180
        out = math.cos(a) * cut + math.sin(a) * out

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, d, depth, widen_factor, num_classes, mix_angle=45):
        super().__init__()
        self.in_planes = 16

        assert (depth % 6 == 4), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k]
        block = functools.partial(wide_basic, mix_angle=mix_angle)

        self.conv1 = conv(d, nStages[0], kernel_size=3, stride=1, padding=1, bias=True, gain=1)
        self.layer1 = self._wide_layer(block, nStages[1], n, stride=1)
        self.layer2 = self._wide_layer(block, nStages[2], n, stride=2)
        self.layer3 = self._wide_layer(block, nStages[3], n, stride=2)
        self.linear = nn.Parameter(torch.randn(num_classes, nStages[3]))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def _wide_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(out)
        out = out.flatten(2).mean(2)

        f = 1 / self.linear.size(1) ** 0.5
        out = F.linear(out, f * self.linear, self.bias)

        if out.size(1) == 1:
            out = out.flatten(0)

        return out
