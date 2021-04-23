import argparse
import copy
import math
import os
import pickle
import subprocess
from functools import partial
from time import perf_counter
import itertools

import torch

from arch import init_arch
from dataset import get_binary_dataset


def loglinspace(step, tau, end=None):
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step * (1 - math.exp(-t / tau)))


def loss_func(f, y, **args):
    if args['loss'] == 'exp':
        return (args['loss_margin'] - args['alpha'] * f * y).exp() / args['alpha']
    if args['loss'] == 'hinge':
        return (args['loss_margin'] - args['alpha'] * f * y).relu() / args['alpha']
    if args['loss'] == 'softhinge':
        sp = partial(torch.nn.functional.softplus, beta=args['loss_beta'])
        return sp(args['loss_margin'] - args['alpha'] * f * y) / args['alpha']
    if args['loss'] == 'qhinge':
        return 0.5 * (args['loss_margin'] - args['alpha'] * f * y).relu().pow(2) / args['alpha']


def sgd_dynamics(f_init, xtr, ytr, out0=None, dt=None, bs=None, replacement=False, **args):
    f = copy.deepcopy(f_init)

    gen = torch.Generator(device="cpu").manual_seed(args['seed_batch'])

    for step in itertools.count():
        if replacement:
            i = torch.randint(len(xtr), size=(bs,), generator=gen)
        else:
            i = torch.randperm(len(xtr), generator=gen)[:bs]

        x = xtr[i]
        y = ytr[i]

        if out0 is None:
            o0 = 0
        else:
            o0 = out0[i]

        oba = f(x) - o0

        lo = loss_func(oba, y, **args)

        f.zero_grad()
        lo.mean().backward()

        state = {
            'step': step,
            't': dt * step,
            'dt': dt,
        }
        internals = {
            'f': f,
            'oba': oba,
            'gradient': torch.cat([p.grad.flatten() for p in f.parameters()])
        }

        yield state, internals

        with torch.no_grad():
            for p in f.parameters():
                p.add_(-dt * p.grad)


def run_sgd(f_init, xtr, ytr, xte, yte, **args):

    with torch.no_grad():
        ote0 = f_init(xte)
        otr0 = f_init(xtr)

    if args['subf0'] == 0:
        ote0 = torch.zeros_like(ote0)
        otr0 = torch.zeros_like(otr0)

    best_test_error = 1
    wall_best_test_error = perf_counter()
    tmp_outputs_index = -1
    margin = 0

    checkpoint_generator = loglinspace(args['ckpt_step'], args['ckpt_tau'])
    checkpoint = next(checkpoint_generator)

    wall = perf_counter()
    dynamics = []
    for state, internals in sgd_dynamics(f_init, xtr, ytr, out0=otr0, **args):
        save_outputs = args['save_outputs']
        save = stop = False
        f = internals['f']

        if state['step'] == checkpoint:
            checkpoint = next(checkpoint_generator)
            save = True
        if not torch.isfinite(internals['oba']).all():
            save = stop = True
        if not torch.isfinite(internals['gradient']).all():
            save = stop = True
        if wall + args['max_wall'] < perf_counter():
            save = save_outputs = stop = True
        if args['wall_max_early_stopping'] is not None and wall_best_test_error + args['wall_max_early_stopping'] < perf_counter():
            save = save_outputs = stop = True

        if not save:
            continue

        state['t_'] = state['dt'] * args['alpha'] / args['h']
        state['grad_norm'] = internals['gradient'].norm().item()
        state['wall'] = perf_counter() - wall
        state['norm'] = sum(p.norm().pow(2) for p in f.parameters()).sqrt().item()
        state['dnorm'] = sum((p0 - p).norm().pow(2) for p0, p in zip(f_init.parameters(), f.parameters())).sqrt().item()

        with torch.no_grad():
            otr = f(xtr) - otr0

        mind = (args['alpha'] * otr * ytr).min().item()
        if mind > margin:
            margin += 0.5
            save = save_outputs = True
        if mind > args['stop_margin']:
            save = save_outputs = stop = True

        with torch.no_grad():
            ote = f(xte) - ote0

        test_err = ((ote * yte <= 0) | ~torch.isfinite(ote)).double().mean().item()
        if test_err < best_test_error:
            if tmp_outputs_index != -1:
                dynamics[tmp_outputs_index]['train']['outputs'] = None
                dynamics[tmp_outputs_index]['train']['labels'] = None
                dynamics[tmp_outputs_index]['test']['outputs'] = None
                dynamics[tmp_outputs_index]['test']['labels'] = None

            best_test_error = test_err
            wall_best_test_error = perf_counter()
            if not save_outputs:
                tmp_outputs_index = len(dynamics)
                save_outputs = True

        if args['arch'] == 'fc':
            def getw(f, i):
                return torch.cat(list(getattr(f.f, "W{}".format(i))))
            state['wnorm'] = [getw(f, i).norm().item() for i in range(f.f.L + 1)]
            state['dwnorm'] = [(getw(f, i) - getw(f_init, i)).norm().item() for i in range(f.f.L + 1)]
            if args['save_weights']:
                assert args['L'] == 1
                W = [getw(f, i) for i in range(2)]
                W0 = [getw(f_init, i) for i in range(2)]
                state['w'] = [W[0][:, j].pow(2).mean().sqrt().item() for j in range(args['d'])]
                state['dw'] = [(W[0][:, j] - W0[0][:, j]).pow(2).mean().sqrt().item() for j in range(args['d'])]
                state['beta'] = W[1].pow(2).mean().sqrt().item()
                state['dbeta'] = (W[1] - W0[1]).pow(2).mean().sqrt().item()
                if args['bias']:
                    B = getattr(f.f, "B0")
                    B0 = getattr(f_init.f, "B0")
                    state['b'] = B.pow(2).mean().sqrt().item()
                    state['db'] = (B - B0).pow(2).mean().sqrt().item()

        state['state'] = copy.deepcopy(f.state_dict()) if save_outputs and (args['save_state'] == 1) else None
        state['train'] = {
            'loss': loss_func(otr, ytr, **args).mean().item(),
            'aloss': args['alpha'] * loss_func(otr, ytr, **args).mean().item(),
            'err': ((otr * ytr <= 0) | ~torch.isfinite(otr)).double().mean().item(),
            'nd': (args['alpha'] * otr * ytr < args['stop_margin']).long().sum().item(),
            'mind': (args['alpha'] * otr * ytr).min().item(),
            'maxd': (args['alpha'] * otr * ytr).max().item(),
            'dfnorm': otr.pow(2).mean().sqrt().item(),
            'fnorm': (otr + otr0).pow(2).mean().sqrt().item(),
            'outputs': otr.cpu().clone() if save_outputs else None,
            'labels': ytr.cpu().clone() if save_outputs else None,
        }
        state['test'] = {
            'loss': loss_func(ote, yte, **args).mean().item(),
            'aloss': args['alpha'] * loss_func(ote, yte, **args).mean().item(),
            'err': test_err,
            'nd': (args['alpha'] * ote * yte < args['stop_margin']).long().sum().item(),
            'mind': (args['alpha'] * ote * yte).min().item(),
            'maxd': (args['alpha'] * ote * yte).max().item(),
            'dfnorm': ote.pow(2).mean().sqrt().item(),
            'fnorm': (ote + ote0).pow(2).mean().sqrt().item(),
            'outputs': ote.cpu().clone() if save_outputs else None,
            'labels': yte.cpu().clone() if save_outputs else None,
        }
        print(
            (
                "[i={d[step]:d} t={d[t]:.2e} wall={d[wall]:.0f}] " + \
                "[train aL={d[train][aloss]:.2e} err={d[train][err]:.2f} " + \
                "nd={d[train][nd]}/{p} mind={d[train][mind]:.3f}] " + \
                "[test aL={d[test][aloss]:.2e} err={d[test][err]:.2f}]"
            ).format(d=state, p=len(ytr)),
            flush=True
        )

        dynamics.append(state)

        out = {
            'dynamics': dynamics,
        }

        yield f, out
        if stop:
            break


def run_exp(f0, xtr, ytr, xte, yte, **args):
    run = {
        'args': args,
        'N': sum(p.numel() for p in f0.parameters()),
        'finished': False,
    }
    wall = None

    for f, out in run_sgd(f0, xtr, ytr, xte, yte, **args):
        run['sgd'] = out

        if wall is None or perf_counter() - wall > 120:
            wall = perf_counter()
            yield run
    yield run

    run['finished'] = True
    yield run


def init(**args):
    torch.backends.cudnn.benchmark = True
    if args['dtype'] == 'float64':
        torch.set_default_dtype(torch.float64)
    if args['dtype'] == 'float32':
        torch.set_default_dtype(torch.float32)

    [(xte, yte, ite), (xtr, ytr, itr)] = get_binary_dataset(
        args['dataset'],
        (args['pte'], args['ptr']),
        (args['seed_testset'] + args['pte'], args['seed_trainset'] + args['ptr']),
        args['d'],
        dtype=torch.get_default_dtype(),
        device=args['device'],
    )

    f, (xtr, xte) = init_arch((xtr, xte), **args)

    return f, xtr, ytr, itr, xte, yte, ite


def execute(**args):
    f, xtr, ytr, itr, xte, yte, ite = init(**args)

    torch.manual_seed(0)  # run_sgd use its own generator
    for run in run_exp(f, xtr, ytr, xte, yte, **args):
        run['dataset'] = {
            'test': ite.cpu().clone() if ite is not None else None,
            'train': itr.cpu().clone() if ite is not None else None,
        }
        yield run


def main():
    git = {
        'log': subprocess.getoutput('git log --format="%H" -n 1 -z'),
        'status': subprocess.getoutput('git status -z'),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default='float64')

    parser.add_argument("--seed_init", type=int, default=0)
    parser.add_argument("--seed_testset", type=int, default=0)
    parser.add_argument("--seed_trainset", type=int, default=0)
    parser.add_argument("--seed_batch", type=int, default=0)

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ptr", type=int, required=True)
    parser.add_argument("--pte", type=int, required=True)
    parser.add_argument("--d", type=int)

    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--act", type=str, required=True)
    parser.add_argument("--act_beta", type=float, default=1.0)
    parser.add_argument("--bias", type=float, default=0)
    parser.add_argument("--last_bias", type=float, default=0)
    parser.add_argument("--var_bias", type=float, default=0)
    parser.add_argument("--L", type=int)
    parser.add_argument("--h", type=int, required=True)
    parser.add_argument("--mix_angle", type=float, default=45)
    parser.add_argument("--cv_L1", type=int, default=2)
    parser.add_argument("--cv_L2", type=int, default=2)
    parser.add_argument("--cv_h_base", type=float, default=1)
    parser.add_argument("--cv_fsz", type=int, default=5)
    parser.add_argument("--cv_pad", type=int, default=1)
    parser.add_argument("--cv_stride_first", type=int, default=1)

    parser.add_argument("--save_outputs", type=int, default=0)
    parser.add_argument("--save_state", type=int, default=0)
    parser.add_argument("--save_weights", type=int, default=0)

    parser.add_argument("--alpha", type=float)
    parser.add_argument("--alpha_", type=float)
    parser.add_argument("--subf0", type=int, default=1)

    parser.add_argument("--bs", type=int, required=True)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--dt_", type=float)
    parser.add_argument("--replacement", type=int, default=0)

    parser.add_argument("--max_wall", type=float, required=True)
    parser.add_argument("--wall_max_early_stopping", type=float)
    parser.add_argument("--chunk", type=int)

    parser.add_argument("--loss", type=str, default="softhinge")
    parser.add_argument("--loss_beta", type=float, default=20.0)
    parser.add_argument("--loss_margin", type=float, default=1.0)
    parser.add_argument("--stop_margin", type=float, default=1.0)

    parser.add_argument("--ckpt_step", type=int, default=100)
    parser.add_argument("--ckpt_tau", type=float, default=1e4)

    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args().__dict__

    if args['device'] is None:
        if torch.cuda.is_available():
            args['device'] = 'cuda'
        else:
            args['device'] = 'cpu'

    if args['chunk'] is None:
        args['chunk'] = max(args['ptr'], args['pte'], 100000)

    if args['seed_init'] == -1:
        args['seed_init'] = args['seed_trainset']

    if args['alpha_'] is None:
        args['alpha_'] = args['alpha'] / args['h']**0.5
    else:
        args['alpha'] = args['alpha_'] * args['h']**0.5

    if args['dt_'] is None:
        args['dt_'] = args['dt'] * args['alpha'] / args['h']
    else:
        args['dt'] = args['dt_'] / args['alpha'] * args['h']

    with open(args['output'], 'wb') as handle:
        pickle.dump(args,  handle)

    saved = False
    try:
        for data in execute(**args):
            data['git'] = git
            with open(args['output'], 'wb') as handle:
                pickle.dump(args, handle)
                pickle.dump(data, handle)
            saved = True
    except:
        if not saved:
            os.remove(args['output'])
        raise


if __name__ == "__main__":
    main()
