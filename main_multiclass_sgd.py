# pylint: disable=C, R, bare-except, arguments-differ, no-member, undefined-loop-variable
import argparse
import copy
import itertools
import os
import subprocess
from functools import partial
from time import perf_counter

import torch

from arch import FC
from arch.swish import swish
from dataset import get_dataset
from dynamics import ContinuousMomentum, gradient, loglinspace, make_step


def output_gradient(f, loss, x, y, out0, bs):
    i = torch.randperm(len(x))[:bs]
    o = f(x[i])
    l = loss(o - out0[i], y[i]).mean()
    grad = gradient(l, f.parameters())
    return o, grad


def train_regular(f0, x, y, tau, loss, subf0, lr, bs):
    f = copy.deepcopy(f0)

    with torch.no_grad():
        with torch.no_grad():
            out0 = f0(x)
        if not subf0:
            out0 = torch.zeros_like(out0)

    optimizer = ContinuousMomentum(f.parameters(), dt=lr, tau=tau)

    t = 0

    out, grad = output_gradient(f, loss, x, y, out0, bs)

    for step in itertools.count():

        state = {
            'step': step,
            't': t,
            'dt': lr,
        }

        yield state, f, out, out0, grad

        if torch.isnan(out).any():
            break

        # make 1 step:

        state = copy.deepcopy((f.state_dict(), optimizer.state_dict(), t))

        make_step(f, optimizer, lr, grad)
        t += lr

        out, grad = output_gradient(f, loss, x, y, out0, bs)


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
    tmp_outputs_index = -1

    torch.manual_seed(args.seed_batch)

    checkpoint_generator = loglinspace(100, 100 * 100)
    checkpoint = next(checkpoint_generator)

    wall = perf_counter()
    dynamics = []
    for state, f, _otr, otr0, grad in train_regular(f0, xtr, ytr, tau, partial(loss_func, args), bool(args.f0), args.lr, args.bs):
        save = False
        save_outputs = False
        stop = False

        if state['step'] == checkpoint:
            checkpoint = next(checkpoint_generator)
            save = True
        if torch.isnan(_otr).any():
            save = True
            stop = True

        if not save:
            continue

        with torch.no_grad():
            otr = f(xtr) - otr0
            ote = f(xte) - ote0

        if args.save_outputs:
            save_outputs = True

        if (otr.argmax(1) != ytr).sum() == 0:
            save_outputs = True
            stop = True

        if wall + args.train_time < perf_counter():
            save_outputs = True
            stop = True

        test_err = (ote.argmax(1) != yte).double().mean().item()
        if test_err < best_test_error:
            if tmp_outputs_index != -1:
                dynamics[tmp_outputs_index]['train']['outputs'] = None
                dynamics[tmp_outputs_index]['train']['labels'] = None
                dynamics[tmp_outputs_index]['test']['outputs'] = None
                dynamics[tmp_outputs_index]['test']['labels'] = None

            best_test_error = test_err
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
                + "[train aL={d[train][aloss]:.2e} err={d[train][err]:.2f}] "
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

        for _f, out in run_regular(args, f0, xtr, ytr, xte, yte):
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
        args.device,
        torch.get_default_dtype()
    )

    torch.manual_seed(0)

    if args.act == 'relu':
        def act(x):
            return torch.relu(x).mul(2 ** 0.5)
    elif args.act == 'tanh':
        def act(x):
            return torch.tanh(x).mul(1.5927116424039378)
    elif args.act == 'softplus':
        factor = torch.nn.functional.softplus(torch.randn(100000, dtype=torch.float64), args.act_beta).pow(2).mean().rsqrt().item()

        def act(x):
            return torch.nn.functional.softplus(x, beta=args.act_beta).mul(factor)
    elif args.act == 'swish':
        act = swish
    else:
        raise ValueError('act not specified')

    _d = abs(act(torch.randn(100000, dtype=torch.float64)).pow(2).mean().rsqrt().item() - 1)
    assert _d < 1e-2, _d

    torch.manual_seed(args.seed_init + hash(args.alpha) + args.ptr)

    c = len(ytr.unique())

    if args.arch == 'fc':
        assert args.L is not None
        xtr = xtr.flatten(1)
        xte = xte.flatten(1)
        f = FC(xtr.size(1), args.h, c, args.L, act, args.bias)
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
    print('deprecated')
    return

    git = {
        'log': subprocess.getoutput('git log --format="%H" -n 1 -z'),
        'status': subprocess.getoutput('git status -z'),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--dtype", type=str, default='float64')

    parser.add_argument("--seed_init", type=int, default=0)
    parser.add_argument("--seed_batch", type=int, default=0)
    parser.add_argument("--seed_testset", type=int, default=0, help="determines the testset, will affect the trainset as well")
    parser.add_argument("--seed_trainset", type=int, default=0, help="determines the trainset")

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ptr", type=int, required=True)
    parser.add_argument("--pte", type=int)
    parser.add_argument("--d", type=int)

    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--act", type=str, required=True)
    parser.add_argument("--act_beta", type=float, default=5.0)
    parser.add_argument("--bias", type=float, default=0)
    parser.add_argument("--L", type=int)
    parser.add_argument("--h", type=int, required=True)

    parser.add_argument("--regular", type=int, default=1)
    parser.add_argument("--save_outputs", type=int, default=0)
    parser.add_argument("--save_state", type=int, default=0)

    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--f0", type=int, default=1)

    parser.add_argument("--tau_over_h", type=float, default=0.0)
    parser.add_argument("--tau_alpha_crit", type=float)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--bs", type=int, required=True)

    parser.add_argument("--train_time", type=float, required=True)
    parser.add_argument("--chunk", type=int)

    parser.add_argument("--loss", type=str, default="crossentropy")

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
