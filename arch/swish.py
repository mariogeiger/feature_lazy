# pylint: disable=no-member, arguments-differ, unused-argument, missing-docstring, invalid-name
import torch
import torch.nn as nn


@torch.jit.script
def _swish_jit_fwd(x):
    return x.mul(torch.sigmoid(x)).mul(1.6768)


@torch.jit.script
def _swish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    return grad_output * (x_sigmoid * (1 + x * (1 - x_sigmoid))) * 1.6768


class _SwishJitAutoFn(torch.autograd.Function):
    """ torch.jit.script optimised Swish
    Inspired by conversation btw Jeremy Howard & Adam Pazske
    https://twitter.com/jeremyphoward/status/1188251041835315200
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return _swish_jit_bwd(x, grad_output)


def swish(x, inplace=False):
    # inplace ignored
    return _SwishJitAutoFn.apply(x)


class Swish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return _SwishJitAutoFn.apply(x)
