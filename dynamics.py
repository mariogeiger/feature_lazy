# pylint: disable=E1101, C
"""
This file implements a continuous version of momentum SGD

It contains two implementation of the same dynamics:
1. `train_regular` for any kind of models
2. `train_kernel` only for linear models
"""
import copy
import itertools
import math
from time import perf_counter

import torch


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


def batch(f, x, y, out0, size, alpha, chunk):
    if size >= len(x):
        return x, y, out0, 0

    with torch.no_grad():
        perm = torch.randperm(len(x), device=x.device)
        xbs = []
        out0bs = []
        ybs = []
        for i in range(0, len(perm), chunk):
            im = perm[i: i + chunk]

            idx = alpha * (f(x[im]) - out0[im]) * y[im] < 1

            xbs.append(x[im][idx])
            out0bs.append(out0[im][idx])
            ybs.append(y[im][idx])
            if sum(len(x) for x in xbs) >= size:
                break
        xb = torch.cat(xbs)[:size]
        out0b = torch.cat(out0bs)[:size]
        yb = torch.cat(ybs)[:size]

    return xb, yb, out0b, i


def train_regular(f0, x, y, temperature, tau, train_time, alpha, chunk, op=None, changes_bounds=(1e-4, 1e-2)):
    f = copy.deepcopy(f0)

    with torch.no_grad():
        out0 = torch.cat([f0(x[i: i + chunk]) for i in range(0, len(x), chunk)])

    if temperature > 0:
        max_dt = chunk * temperature / (1 - (chunk - 1) / (len(x) - 1)) if chunk < len(x) else math.inf
        dt = temperature
    else:
        max_dt = math.inf
        dt = 1e-50
    step_change_dt = 0
    optimizer = ContinuousMomentum(f.parameters(), dt=dt, tau=tau)

    dynamics = []
    checkpoint_generator = loglinspace(0.05, 100)
    checkpoint = next(checkpoint_generator)
    time = perf_counter()
    t = 0
    out_new = None

    for step in itertools.count():

        while True:
            bs = len(x) / (temperature / dt * (len(x) - 1) + 1)
            bs = int(round(bs))

            if bs == 0:
                bs = 1

            xb, yb, out0b, i = batch(f, x, y, out0, bs, alpha, chunk)
            if len(xb) == 0:
                break

            if temperature == 0 and out_new is not None:
                out = out_new
            else:
                out = f(xb)
            loss = (1 - alpha * (out - out0b) * yb).relu().sum() / bs / alpha ** 2

            if temperature == 0 and loss.item() == 0:
                break

            optimizer.zero_grad()
            loss.backward()

            state = copy.deepcopy((f.state_dict(), optimizer.state_dict(), t))
            optimizer.step()
            t += dt
            current_dt = dt

            if temperature == 0:
                out_new = f(xb)
            else:
                with torch.no_grad():
                    out_new = f(xb)

            df = alpha * (out - out_new).abs().max().item()
            if math.isnan(df):
                break

            if changes_bounds[0] < df < changes_bounds[1]:
                break

            if df < changes_bounds[1] and dt >= max_dt:
                break

            if df > changes_bounds[1]:
                f.load_state_dict(state[0])
                optimizer.load_state_dict(state[1])
                t = state[2]
                out_new = None
                dt /= 10
                for param_group in optimizer.param_groups:
                    param_group['dt'] = dt
                print("[{} +{}] max df={:.1e}".format(step, step - step_change_dt, df), flush=True)

            step_change_dt = step

            if df < changes_bounds[0]:
                if df == 0:
                    dt = min(10 * dt, max_dt)
                else:
                    dt = min(1.1 * dt, max_dt)
                for param_group in optimizer.param_groups:
                    param_group['dt'] = dt
                break


        if len(xb) == 0:
            break
        if temperature == 0 and loss.item() == 0:
            break

        if step == checkpoint:
            checkpoint = next(checkpoint_generator)
            assert checkpoint > step

            state = {
                'step': step,
                'time': perf_counter() - time,
                't': t,
                'dt': current_dt,
                'bs': bs,
                'df': df,
                'batch_loss': loss.item(),
                'ibs': i / len(x),
                'norm': sum(p.norm().pow(2) for p in f.parameters()).sqrt().item(),
                'dnorm': sum((p0 - p).norm().pow(2) for p0, p in zip(f0.parameters(), f.parameters())).sqrt().item(),
            }

            if op is not None:
                state = op(f, state)
            else:
                print("[{d[step]:d} t={d[t]:.2e} wall={d[time]:.0f}] [dt={d[dt]:.1e} bs={d[bs]:d} df={d[df]:.1e}] [train i/P={d[ibs]:.2f} L={d[batch_loss]:.2e}]".format(d=state), flush=True)

            dynamics.append(state)

        if perf_counter() > time + train_time:
            break

        if torch.isnan(out).any():
            break

    return f, dynamics


def train_kernel(ktrtr, ytr, temperature, tau, train_time, alpha, changes_bounds=(1e-4, 1e-2)):
    # (1 - a f y).relu / a^2  =>  -y theta(1 - a f y) / a
    otr = ktrtr.new_zeros(len(ytr))
    velo = otr.clone()

    if temperature > 0:
        dt = temperature
    else:
        dt = 1e-50
    step_change_dt = 0

    dynamics = []
    checkpoint_generator = loglinspace(0.05, 1000)
    checkpoint = next(checkpoint_generator)
    time = perf_counter()
    t = 0

    for step in itertools.count():

        while True:
            bs = len(otr) / (temperature / dt * (len(otr) - 1) + 1)
            bs = int(round(bs))

            if bs == 0:
                bs = 1

            lprim = -ytr * (1 - alpha * otr * ytr >= 0).type(otr.dtype) / alpha
            B = lprim.nonzero().flatten()
            B = B[torch.randperm(len(B))[:bs]]

            if len(B) == 0:
                break

            grad = ktrtr[:, B] @ lprim[B] / bs

            state = copy.deepcopy((otr, velo, t))
            if t == 0:
                velo.zero_()
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

            df = alpha * dt * velo.abs().max().item()
            if math.isnan(df):
                break

            if changes_bounds[0] < df < changes_bounds[1]:
                break

            if df > changes_bounds[1]:
                otr, velo, t = state
                dt /= 10
                print("[{} +{}] max df={:.1e}".format(step, step - step_change_dt, df), flush=True)

            step_change_dt = step

            if df < changes_bounds[0]:
                dt *= 1.1 if df > 0 else 10
                break


        if len(B) == 0:
            break

        if step == checkpoint:
            checkpoint = next(checkpoint_generator)
            assert checkpoint > step

            dynamics.append({
                'step': step,
                'time': perf_counter() - time,
                't': t,
                'dt': current_dt,
                'bs': bs,
                'df': df,
                'err': (otr * ytr <= 0).double().mean().item(),
                'nd': (alpha * otr * ytr < 1).long().sum().item(),
                'loss': alpha ** -2 * (1 - alpha * otr * ytr).relu().mean().item(),
            })
            print("[{d[step]:d} t={d[t]:.2e} wall={d[time]:.0f}] [dt={d[dt]:.1e} bs={d[bs]:d} df={d[df]:.1e}] [train L={d[loss]:.2e} err={d[err]:.2f} nd={d[nd]}]".format(d=dynamics[-1]), flush=True)

        if perf_counter() > time + train_time:
            break

        if torch.isnan(otr).any():
            break

    return otr, dynamics
