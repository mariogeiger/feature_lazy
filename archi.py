# pylint: disable=E1101, C, arguments-differ
import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CV(nn.Module):
    def __init__(self, d, h, L1, L2, act, h_base, fsz, sig_w, sig_b, beta, pad, stride_first, split_w=False):
        super().__init__()

        f1 = d
        for i in range(L1):
            f2 = int(h * h_base ** i)
            for j in range(L2):
                if split_w:
                    W = nn.ParameterList([nn.Parameter(torch.randn(f1, fsz, fsz)) for _ in range(f2)])
                    setattr(self, "W{}_{}".format(i, j), W)
                else:
                    self.register_parameter("W{}_{}".format(i, j), nn.Parameter(torch.randn(f2, f1, fsz, fsz)))

                self.register_parameter("b{}_{}".format(i, j), nn.Parameter(sig_b * torch.randn(f2)))
                f1 = f2

        self.register_parameter("W", nn.Parameter(torch.randn(f1)))
        self.register_parameter("b", nn.Parameter(sig_b * torch.randn(())))

        self.L1 = L1
        self.L2 = L2
        self.act = act
        self.sig_w = sig_w
        self.beta = beta
        self.pad = pad
        self.stride_first = stride_first

    def forward(self, x):
        for i in range(self.L1):
            assert x.size(2) >= 5 and x.size(3) >= 5
            for j in range(self.L2):
                W = getattr(self, "W{}_{}".format(i, j))
                b = getattr(self, "b{}_{}".format(i, j))

                if isinstance(W, nn.ParameterList):
                    W = torch.stack(list(W))

                stride = 2 if j == 0 and (i > 0 or self.stride_first) else 1
                f = W[0].numel()
                x = nn.functional.conv2d(x, W * (self.sig_w / f ** 0.5), self.beta * b, stride=stride, padding=self.pad)
                x = self.act(x)

        x = x.flatten(2).mean(2)

        W = getattr(self, "W")
        b = getattr(self, "b")
        f = len(W)
        x = x @ (W * (self.sig_w / f ** 0.5)) + b
        return x.view(-1)


def mnist_cv(h, d=1, split_w=False):
    return CV(d, h, 2, 2, torch.relu, 1, 5, 2**0.5, 0, 1, 1, True, split_w)


class FC(nn.Module):
    def __init__(self, d, h, L, act, sig_w, sig_b):
        super().__init__()

        f = d
        for i in range(L):
            W = torch.randn(h, f)
            n = max(1, 256**2 // f)
            W = nn.ParameterList([nn.Parameter(W[j: j+n]) for j in range(0, len(W), n)])
            setattr(self, "W{}".format(i), W)

            self.register_parameter("b{}".format(i), nn.Parameter(sig_b * torch.randn(h)))
            f = h

        self.register_parameter("W{}".format(L), nn.Parameter(torch.randn(1, f)))
        self.register_parameter("b{}".format(L), nn.Parameter(sig_b * torch.randn(1)))

        self.L = L
        self.act = act
        self.sig_w = sig_w

    def forward(self, x):
        for i in range(self.L + 1):
            W = getattr(self, "W{}".format(i))
            b = getattr(self, "b{}".format(i))

            if isinstance(W, nn.ParameterList):
                W = torch.cat(list(W))

            f = x.size(1)
            x = x @ W.t() * (self.sig_w / f ** 0.5) + b

            if i < self.L:
                x = self.act(x)

        return x.view(-1)


class FC_AVG(nn.Module):
    def __init__(self, d, h, L, act, sig_w, sig_b, n_avg):
        super().__init__()

        fi = d
        for i in range(L + 1):
            fo = h
            if n_avg is not None and i == L:
                fo = n_avg

            W = torch.randn(fo, fi)
            n = max(1, 256**2 // fi)
            W = nn.ParameterList([nn.Parameter(W[j: j+n]) for j in range(0, len(W), n)])
            setattr(self, "W{}".format(i), W)

            self.register_parameter("b{}".format(i), nn.Parameter(sig_b * torch.randn(fo)))
            fi = fo

        self.L = L
        self.act = act
        self.sig_w = sig_w

    def forward(self, x):
        for i in range(self.L + 1):
            W = getattr(self, "W{}".format(i))
            b = getattr(self, "b{}".format(i))

            W = torch.cat(list(W))

            f = x.size(1)
            x = x @ W.t() * (self.sig_w / f ** 0.5) + b

            if i < self.L:
                x = self.act(x)

        return x.mean(1)


class conv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, bias=True, gain=1, init_=nn.init.normal_):
        super().__init__()

        w = init_(torch.empty(out_planes, in_planes, kernel_size, kernel_size))
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
    def __init__(self, in_planes, planes, stride=1, mix_angle=45, init_=nn.init.normal_):
        super().__init__()
        self.conv1 = conv(in_planes, planes, kernel_size=3, padding=1, bias=True, gain=2 ** 0.5, init_=init_)
        self.conv2 = conv(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True, gain=2 ** 0.5, init_=init_)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = conv(in_planes, planes, kernel_size=1, stride=stride, bias=True, gain=1, init_=init_)

        self.mix_angle = mix_angle

    def forward(self, x):
        out = self.conv1(F.relu(x))
        out = self.conv2(F.relu(out))
        cut = self.shortcut(x)

        a = self.mix_angle * math.pi / 180
        out = math.cos(a) * cut + math.sin(a) * out

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, d, depth, widen_factor, num_classes, mix_angle=45, init_=nn.init.normal_):
        super().__init__()
        self.in_planes = 16

        assert (depth % 6 == 4), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k]
        block = functools.partial(wide_basic, mix_angle=mix_angle, init_=init_)

        self.conv1 = conv(d, nStages[0], kernel_size=3, stride=1, padding=1, bias=True, gain=1, init_=init_)
        self.layer1 = self._wide_layer(block, nStages[1], n, stride=1)
        self.layer2 = self._wide_layer(block, nStages[2], n, stride=2)
        self.layer3 = self._wide_layer(block, nStages[3], n, stride=2)
        self.linear = nn.Parameter(init_(torch.empty(num_classes, nStages[3])))
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


def normal_orthogonal_(tensor, gain=1):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new_empty(rows, cols).normal_(0, 1)

    for i in range(0, rows, cols):
        # Compute the qr factorization
        q, r = torch.qr(flattened[i:i + cols].t())
        # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
        q *= torch.diag(r, 0).sign()
        q.t_()

        with torch.no_grad():
            tensor[i:i + cols].view_as(q).copy_(q)

    with torch.no_grad():
        tensor.mul_(cols ** 0.5 * gain)
    return tensor
