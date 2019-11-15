# pylint: disable=no-member, arguments-differ, unused-argument, missing-docstring, invalid-name
import torch
import torch.nn as nn


@torch.jit.script
def swish_jit_fwd(x):
    return x.mul(torch.sigmoid(x))


@torch.jit.script
def swish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    return grad_output * (x_sigmoid * (1 + x * (1 - x_sigmoid)))


class SwishJitAutoFn(torch.autograd.Function):
    """ torch.jit.script optimised Swish
    Inspired by conversation btw Jeremy Howard & Adam Pazske
    https://twitter.com/jeremyphoward/status/1188251041835315200
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        return swish_jit_bwd(x, grad_output)


def swish_jit(x, inplace=False):
    # inplace ignored
    return SwishJitAutoFn.apply(x)


class SwishJit(nn.Module):
    def __init__(self, inplace: bool = False):
        super(SwishJit, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return SwishJitAutoFn.apply(x)
