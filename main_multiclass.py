# pylint: disable=C, R, bare-except, arguments-differ, no-member, undefined-loop-variable
import argparse
import copy
import os
import subprocess
from functools import partial
from time import perf_counter

import torch

from arch import FC
from arch.mnas import MnasNetLike
from arch.swish import swish
from dataset import get_dataset
from dynamics import train_regular, loglinspace


def loss_func(args, f, y):
    if args.loss == 'crossentropy':
        return torch.nn.functional.cross_entropy(args.alpha * f, y, reduction='none') / args.alpha


class SplitEval(torch.nn.Module):
    def __init__(self, f, size):
        super().__init__()
        self.f = f
        self.size = size

    def forward(self, x):
        return torch.cat([self.f(x[i: i + self.size]) for i in range(0, len(x), self.size)])


def run_regular(args, f0, xtr, ytr, xte, yte):

    with torch.no_grad():
        ote0 = f0(xte)

    if args.f0 == 0:
        ote0 = torch.zeros_like(ote0)

    tau = args.tau_over_h * args.h
    if args.tau_alpha_crit is not None:
        tau *= min(1, args.tau_alpha_crit / args.alpha)

    best_test_error = 1
    wall_best_test_error = perf_counter()
    tmp_outputs_index = -1

    checkpoint_generator = loglinspace(args.ckpt_step, args.ckpt_tau)
    checkpoint = next(checkpoint_generator)

    wall = perf_counter()
    dynamics = []
    for state, f, otr, otr0, grad in train_regular(f0, xtr, ytr, tau, args.alpha, partial(loss_func, args), bool(args.f0), args.chunk, args.max_dgrad, args.max_dout):
        otr = otr - otr0

        save_outputs = args.save_outputs
        save = stop = False

        if state['step'] == checkpoint:
            checkpoint = next(checkpoint_generator)
            save = True
        if torch.isnan(otr).any():
            save = stop = True
        if wall + args.train_time < perf_counter():
            save = save_outputs = stop = True
        if args.wall_max_early_stopping is not None and wall_best_test_error + args.wall_max_early_stopping < perf_counter():
            save = save_outputs = stop = True
        if (otr.argmax(1) != ytr).sum() == 0:
            save = save_outputs = stop = True

        if not save:
            continue

        with torch.no_grad():
            ote = f(xte) - ote0

        test_err = (ote.argmax(1) != yte).double().mean().item()
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

        state['grad_norm'] = grad.norm().item()
        state['wall'] = perf_counter() - wall
        state['norm'] = sum(p.norm().pow(2) for p in f.parameters()).sqrt().item()
        state['dnorm'] = sum((p0 - p).norm().pow(2) for p0, p in zip(f0.parameters(), f.parameters())).sqrt().item()

        if args.arch == 'fc':
            def getw(f, i):
                return torch.cat(list(getattr(f.f, "W{}".format(i))))
            state['wnorm'] = [getw(f, i).norm().item() for i in range(f.f.L + 1)]
            state['dwnorm'] = [(getw(f, i) - getw(f0, i)).norm().item() for i in range(f.f.L + 1)]

        state['state'] = copy.deepcopy(f.state_dict()) if save_outputs and (args.save_state == 1) else None
        state['train'] = {
            'loss': loss_func(args, otr, ytr).mean().item(),
            'aloss': args.alpha * loss_func(args, otr, ytr).mean().item(),
            'err': (otr.argmax(1) != ytr).double().mean().item(),
            'dfnorm': otr.pow(2).mean().sqrt(),
            'fnorm': (otr + otr0).pow(2).mean().sqrt(),
            'outputs': otr if save_outputs else None,
            'labels': ytr if save_outputs else None,
        }
        state['test'] = {
            'loss': loss_func(args, ote, yte).mean().item(),
            'aloss': args.alpha * loss_func(args, ote, yte).mean().item(),
            'err': test_err,
            'dfnorm': ote.pow(2).mean().sqrt(),
            'fnorm': (ote + ote0).pow(2).mean().sqrt(),
            'outputs': ote if save_outputs else None,
            'labels': yte if save_outputs else None,
        }
        print(
            (
                "[i={d[step]:d} t={d[t]:.2e} wall={d[wall]:.0f}] "
                + "[dt={d[dt]:.1e} dgrad={d[dgrad]:.1e} dout={d[dout]:.1e}] "
                + "[train aL={d[train][aloss]:.2e} err={d[train][err]:.2f} "
                + "] "
                + "[test aL={d[test][aloss]:.2e} err={d[test][err]:.2f}]"
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


def run_exp(args, f0, xtr, ytr, xte, yte):
    run = {
        'args': args,
        'N': sum(p.numel() for p in f0.parameters()),
        'finished': False,
    }

    if args.regular == 1:
        wall = perf_counter()
        for f, out in run_regular(args, f0, xtr, ytr, xte, yte):
            run['regular'] = out

            if perf_counter() - wall > 120:
                wall = perf_counter()
                yield run
        yield run

    run['finished'] = True
    yield run


def init(args):
    torch.backends.cudnn.benchmark = True
    if args.dtype == 'float64':
        torch.set_default_dtype(torch.float64)
    if args.dtype == 'float32':
        torch.set_default_dtype(torch.float32)

    [(xte, yte, ite), (xtr, ytr, itr)] = get_dataset(
        args.dataset,
        (args.pte, args.ptr),
        (args.seed_testset + args.pte, args.seed_trainset + args.ptr),
        args.d,
        None,
        args.device,
        torch.get_default_dtype()
    )

    torch.manual_seed(0)

    if args.act == 'relu':
        _act = torch.relu
    elif args.act == 'tanh':
        _act = torch.tanh
    elif args.act == 'softplus':
        _act = torch.nn.functional.softplus
    elif args.act == 'swish':
        _act = swish
    else:
        raise ValueError('act not specified')

    def __act(x):
        b = args.act_beta
        return _act(b * x) / b
    factor = __act(torch.randn(100000, dtype=torch.float64)).pow(2).mean().rsqrt().item()

    def act(x):
        return __act(x) * factor

    _d = abs(act(torch.randn(100000, dtype=torch.float64)).pow(2).mean().rsqrt().item() - 1)
    assert _d < 1e-2, _d

    torch.manual_seed(args.seed_init + hash(args.alpha) + args.ptr)

    c = len(ytr.unique())

    if args.arch == 'fc':
        assert args.L is not None
        xtr = xtr.flatten(1)
        xte = xte.flatten(1)
        f = FC(xtr.size(1), args.h, c, args.L, act, args.bias, args.var_bias)

    elif args.arch == 'mnas':
        assert args.act == 'swish'
        f = MnasNetLike(xtr.size(1), args.h, c, args.cv_L1, args.cv_L2, dim=xtr.dim() - 2)

    else:
        raise ValueError('arch not specified')

    f = SplitEval(f, args.chunk)
    f = f.to(args.device)

    return f, xtr, ytr, itr, xte, yte, ite


def execute(args):
    f, xtr, ytr, itr, xte, yte, ite = init(args)

    torch.manual_seed(0)
    for run in run_exp(args, f, xtr, ytr, xte, yte):
        run['dataset'] = {
            'test': ite,
            'train': itr,
        }
        yield run


def main():
    git = {
        'log': subprocess.getoutput('git log --format="%H" -n 1 -z'),
        'status': subprocess.getoutput('git status -z'),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--dtype", type=str, default='float64')

    parser.add_argument("--seed_init", type=int, default=0)
    parser.add_argument("--seed_testset", type=int, default=0, help="determines the testset, will affect the trainset as well")
    parser.add_argument("--seed_trainset", type=int, default=0, help="determines the trainset")

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ptr", type=int, required=True)
    parser.add_argument("--pte", type=int)
    parser.add_argument("--d", type=int)

    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--act", type=str, required=True)
    parser.add_argument("--act_beta", type=float, default=1.0)
    parser.add_argument("--bias", type=float, default=0)
    parser.add_argument("--var_bias", type=float, default=0)
    parser.add_argument("--L", type=int)
    parser.add_argument("--h", type=int, required=True)
    parser.add_argument("--cv_L1", type=int, default=2)
    parser.add_argument("--cv_L2", type=int, default=2)

    parser.add_argument("--regular", type=int, default=1)
    parser.add_argument("--save_outputs", type=int, default=0)
    parser.add_argument("--save_state", type=int, default=0)

    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--f0", type=int, default=1)

    parser.add_argument("--tau_over_h", type=float, default=0.0)
    parser.add_argument("--tau_alpha_crit", type=float)

    parser.add_argument("--train_time", type=float, required=True)
    parser.add_argument("--wall_max_early_stopping", type=float)
    parser.add_argument("--chunk", type=int)
    parser.add_argument("--max_dgrad", type=float, default=1e-4)
    parser.add_argument("--max_dout", type=float, default=1e-1)

    parser.add_argument("--loss", type=str, default="crossentropy")
    parser.add_argument("--loss_beta", type=float, default=20.0)
    parser.add_argument("--loss_margin", type=float, default=1.0)

    parser.add_argument("--ckpt_step", type=int, default=100)
    parser.add_argument("--ckpt_tau", type=float, default=1e4)

    parser.add_argument("--pickle", type=str, required=True)
    args = parser.parse_args()

    if args.pte is None:
        args.pte = args.ptr

    if args.chunk is None:
        args.chunk = max(args.ptr, args.pte, args.ptk)

    torch.save(args, args.pickle)
    saved = False
    try:
        for res in execute(args):
            res['git'] = git
            with open(args.pickle, 'wb') as f:
                torch.save(args, f)
                torch.save(res, f)
                saved = True
    except:
        if not saved:
            os.remove(args.pickle)
        raise


if __name__ == "__main__":
    main()
