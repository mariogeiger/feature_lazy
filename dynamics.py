# pylint: disable=E1101, C
"""
This file implements a continuous version of momentum SGD
Dynamics that compares the angle of the gradient between steps and keep it small

- stop when margins are reached

It contains two implementation of the same dynamics:
1. `train_regular` for any kind of models
2. `train_kernel` only for linear models
"""
import copy
import itertools
import math

import torch

from hessian import gradient


def loglinspace(step, tau, end=None):
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step * (1 - math.exp(-t / tau)))


class ContinuousMomentum(torch.optim.Optimizer):
    r"""Implements a continuous version of momentum.

    d/dt velocity = -1/tau (velocity + grad)
     or
    d/dt velocity = -mu/t (velocity + grad)

    d/dt parameters = velocity
    """

    def __init__(self, params, dt, tau):
        defaults = dict(dt=dt, tau=tau)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            tau = group['tau']
            dt = group['dt']

            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]
                if 't' not in param_state:
                    t = param_state['t'] = 0
                else:
                    t = param_state['t']

                if tau != 0:
                    if 'velocity' not in param_state:
                        v = param_state['velocity'] = torch.zeros_like(p.data)
                    else:
                        v = param_state['velocity']

                if tau > 0:
                    x = math.exp(-dt / tau)
                    v.mul_(x).add_(-(1 - x), p.grad.data)
                elif tau < 0:
                    mu = -tau
                    x = (t / (t + dt)) ** mu
                    v.mul_(x).add_(-(1 - x), p.grad.data)
                else:
                    v = -p.grad.data

                p.data.add_(dt, v)
                param_state['t'] += dt

        return loss


def make_step(f, optimizer, dt, grad):
    i = 0
    for p in f.parameters():
        n = p.numel()
        p.grad = grad[i: i + n].view_as(p)
        i += n

    for param_group in optimizer.param_groups:
        param_group['dt'] = dt

    optimizer.step()

    for p in f.parameters():
        p.grad = None


def output_gradient(f, loss, x, y, out0, chunk):
    out = []
    grad = 0
    for i in [slice(i, i + chunk) for i in range(0, len(x), chunk)]:
        o = f(x[i])
        l = loss(o - out0[i], y[i]).sum() / len(x)
        grad += gradient(l, f.parameters())
        out.append(o)
    return torch.cat(out), grad


def train_regular(f0, x, y, tau, alpha, loss, subf0, chunk, max_dgrad=math.inf, max_dout=math.inf):
    f = copy.deepcopy(f0)

    with torch.no_grad():
        with torch.no_grad():
            out0 = f0(x)
        if not subf0:
            out0 = torch.zeros_like(out0)

    dt = 1
    current_dt = 0
    step_change_dt = 0
    optimizer = ContinuousMomentum(f.parameters(), dt=dt, tau=tau)

    t = 0

    out, grad = output_gradient(f, loss, x, y, out0, chunk)
    dgrad, dout = 0, 0

    for step in itertools.count():

        state = {
            'step': step,
            't': t,
            'dt': current_dt,
            'dgrad': dgrad,
            'dout': dout,
        }

        yield state, f, out, out0, grad

        if torch.isnan(out).any():
            break

        # make 1 step:

        state = copy.deepcopy((f.state_dict(), optimizer.state_dict(), t))

        while True:
            make_step(f, optimizer, dt, grad)
            t += dt
            current_dt = dt

            new_out, new_grad = output_gradient(f, loss, x, y, out0, chunk)

            dout = (out - new_out).mul(alpha).abs().max().item()
            if grad.norm() == 0 or new_grad.norm() == 0:
                dgrad = 0
            else:
                dgrad = (grad - new_grad).norm().pow(2).div(grad.norm() * new_grad.norm()).item()

            if dgrad < max_dgrad and dout < max_dout:
                if dgrad < 0.5 * max_dgrad and dout < 0.5 * max_dout:
                    dt *= 1.1
                break

            dt /= 10

            print("[{} +{}] [dt={:.1e} dgrad={:.1e} dout={:.1e}]".format(step, step - step_change_dt, dt, dgrad, dout), flush=True)
            step_change_dt = step
            f.load_state_dict(state[0])
            optimizer.load_state_dict(state[1])
            t = state[2]

        out = new_out
        grad = new_grad


def train_kernel(ktrtr, ytr, tau, alpha, loss_prim, max_dgrad=math.inf, max_dout=math.inf):
    otr = ktrtr.new_zeros(len(ytr))
    velo = otr.clone()

    dt = 1
    step_change_dt = 0

    t = 0
    current_dt = 0

    lprim = loss_prim(otr, ytr)
    grad = ktrtr @ lprim / len(ytr)
    dgrad, dout = 0, 0

    for step in itertools.count():

        state = {
            'step': step,
            't': t,
            'dt': current_dt,
            'dgrad': dgrad,
            'dout': dout,
        }

        yield state, otr, velo, grad

        if torch.isnan(otr).any():
            break

        # make 1 step:

        state = copy.deepcopy((otr, velo, t))

        while True:

            if tau > 0:
                x = math.exp(-dt / tau)
                velo.mul_(x).add_(-(1 - x), grad)
            elif tau < 0:
                mu = -tau
                x = (t / (t + dt)) ** mu
                velo.mul_(x).add_(-(1 - x), grad)
            else:
                velo.copy_(-grad)
            otr.add_(dt, velo)

            t += dt
            current_dt = dt

            lprim = loss_prim(otr, ytr)
            new_grad = ktrtr @ lprim / len(ytr)

            dout = velo.mul(dt * alpha).abs().max().item()
            if grad.norm() == 0 or new_grad.norm() == 0:
                dgrad = 0
            else:
                dgrad = (grad - new_grad).norm().pow(2).div(grad.norm() * new_grad.norm()).item()

            if dgrad < max_dgrad and dout < max_dout:
                if dgrad < 0.1 * max_dgrad and dout < 0.1 * max_dout:
                    dt *= 1.1
                break

            dt /= 10

            print("[{} +{}] [dt={:.1e} dgrad={:.1e} dout={:.1e}]".format(step, step - step_change_dt, dt, dgrad, dout), flush=True)
            step_change_dt = step
            otr.copy_(state[0])
            velo.copy_(state[1])
            t = state[2]

        grad = new_grad
