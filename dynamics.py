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
from time import perf_counter

import torch

from hessian import gradient


def loglinspace(rate, step, end=None):
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step * (1 - math.exp(-t * rate / step)))


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


def train_regular(f0, x, y, tau, max_walltime, alpha, loss, max_dgrad=math.inf, max_dout=math.inf):
    f = copy.deepcopy(f0)

    with torch.no_grad():
        out0 = f0(x)

    dt = 1
    step_change_dt = 0
    optimizer = ContinuousMomentum(f.parameters(), dt=dt, tau=tau)

    checkpoint_generator = loglinspace(0.01, 100)
    checkpoint = next(checkpoint_generator)
    wall = perf_counter()
    t = 0
    converged = False

    out = f(x)
    grad = gradient(loss((out - out0) * y).mean(), f.parameters())

    for step in itertools.count():

        state = copy.deepcopy((f.state_dict(), optimizer.state_dict(), t))

        while True:
            make_step(f, optimizer, dt, grad)
            t += dt
            current_dt = dt

            new_out = f(x)
            new_grad = gradient(loss((new_out - out0) * y).mean(), f.parameters())

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

        save = False

        if step == checkpoint:
            checkpoint = next(checkpoint_generator)
            assert checkpoint > step
            save = True

        if (alpha * (out - out0) * y >= 1).all() and not converged:
            converged = True
            save = True

        if save:
            state = {
                'step': step,
                'wall': perf_counter() - wall,
                't': t,
                'dt': current_dt,
                'dgrad': dgrad,
                'dout': dout,
                'norm': sum(p.norm().pow(2) for p in f.parameters()).sqrt().item(),
                'dnorm': sum((p0 - p).norm().pow(2) for p0, p in zip(f0.parameters(), f.parameters())).sqrt().item(),
                'grad_norm': grad.norm().item(),
            }

            yield f, state, converged

        if converged:
            break

        if perf_counter() > wall + max_walltime:
            break

        if torch.isnan(out).any():
            break



def train_kernel(ktrtr, ytr, tau, max_walltime, alpha, loss_prim, max_dgrad=math.inf, max_dout=math.inf):
    otr = ktrtr.new_zeros(len(ytr))
    velo = otr.clone()

    dt = 1
    step_change_dt = 0

    checkpoint_generator = loglinspace(0.01, 100)
    checkpoint = next(checkpoint_generator)
    wall = perf_counter()
    t = 0
    converged = False

    lprim = loss_prim(otr * ytr) * ytr
    grad = ktrtr @ lprim / len(ytr)

    for step in itertools.count():

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

            lprim = loss_prim(otr * ytr) * ytr
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

        save = False

        if step == checkpoint:
            checkpoint = next(checkpoint_generator)
            assert checkpoint > step
            save = True

        if (alpha * otr * ytr >= 1).all() and not converged:
            converged = True
            save = True

        if save:
            state = {
                'step': step,
                'wall': perf_counter() - wall,
                't': t,
                'dt': current_dt,
                'dgrad': dgrad,
                'dout': dout,
                'grad_norm': grad.norm().item(),
            }

            yield otr, velo, grad, state, converged

        if converged:
            break

        if perf_counter() > wall + max_walltime:
            break

        if torch.isnan(otr).any():
            break
