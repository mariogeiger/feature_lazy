# pylint: disable=C, R, bare-except, arguments-differ, no-member, undefined-loop-variable
import argparse
import math
import os
import subprocess
from functools import partial
from time import perf_counter

import torch

from archi import CV, FC, Wide_ResNet
from dataset import get_binary_dataset, get_binary_pca_dataset
from dynamics import train_kernel, train_regular
from kernels import compute_kernels


def loss_func(args, fy):
    if args.loss == 'softhinge':
        sp = partial(torch.nn.functional.softplus, beta=args.lossbeta)
        return sp(1 - args.alpha * fy) / args.alpha
    if args.loss == 'qhinge':
        return 0.5 * (1 - args.alpha * fy).relu().pow(2) / args.alpha


def loss_func_prime(args, fy):
    if args.loss == 'softhinge':
        return -torch.sigmoid(args.lossbeta * (1 - args.alpha * fy)).mul(args.lossbeta)
    if args.loss == 'qhinge':
        return -(1 - args.alpha * fy).relu()


class SplitEval(torch.nn.Module):
    def __init__(self, f, size):
        super().__init__()
        self.f = f
        self.size = size

    def forward(self, x):
        return torch.cat([self.f(x[i: i + self.size]) for i in range(0, len(x), self.size)])


def run_kernel(args, ktrtr, ktetr, ktete, f, xtr, ytr, xte, yte):
    assert args.f0 == 1

    dynamics = []

    tau = args.tau_over_h * args.h
    if args.tau_alpha_crit is not None:
        tau *= min(1, args.tau_alpha_crit / args.alpha)

    for otr, _velo, _grad, state, _converged in train_kernel(ktrtr, ytr, tau, args.train_time, args.alpha, partial(loss_func_prime, args), args.max_dgrad, args.max_dout):
        state['train'] = {
            'loss': loss_func(args, otr * ytr).mean().item(),
            'aloss': args.alpha * loss_func(args, otr * ytr).mean().item(),
            'err': (otr * ytr <= 0).double().mean().item(),
            'nd': (args.alpha * otr * ytr < 1).long().sum().item(),
            'dfnorm': otr.pow(2).mean().sqrt(),
            'outputs': otr if args.save_outputs else None,
            'labels': ytr if args.save_outputs else None,
        }

        print("[i={d[step]:d} t={d[t]:.2e} wall={d[wall]:.0f}] [dt={d[dt]:.1e} dgrad={d[dgrad]:.1e} dout={d[dout]:.1e}] [train aL={d[train][aloss]:.2e} err={d[train][err]:.2f} nd={d[train][nd]}]".format(d=state), flush=True)
        dynamics.append(state)

    c = torch.lstsq(otr.view(-1, 1), ktrtr).solution.flatten()

    if len(xte) > len(xtr):
        from hessian import gradient
        a = gradient(f(xtr) @ c, f.parameters())
        ote = torch.stack([gradient(f(x[None]), f.parameters()) @ a for x in xte])
    else:
        ote = ktetr @ c

    out = {
        'dynamics': dynamics,
        'train': {
            'outputs': otr,
            'labels': ytr,
        },
        'test': {
            'outputs': ote,
            'labels': yte,
        },
        'kernel': {
            'train': {
                'value': ktrtr.cpu() if args.store_kernel == 1 else None,
                'diag': ktrtr.diag(),
                'mean': ktrtr.mean(),
                'std': ktrtr.std(),
                'norm': ktrtr.norm(),
            },
            'test': {
                'value': ktete.cpu() if args.store_kernel == 1 else None,
                'diag': ktete.diag(),
                'mean': ktete.mean(),
                'std': ktete.std(),
                'norm': ktete.norm(),
            },
        },
    }

    return out


def run_regular(args, f0, xtr, ytr, xte, yte):

    with torch.no_grad():
        otr0 = f0(xtr)
        ote0 = f0(xte)

    if args.f0 == 0:
        otr0 = torch.zeros_like(otr0)
        ote0 = torch.zeros_like(ote0)

    j = torch.randperm(min(len(xte), len(xtr)))[:10 * args.chunk]
    ytrj = ytr[j]
    ytej = yte[j]

    t = perf_counter()

    tau = args.tau_over_h * args.h
    if args.tau_alpha_crit is not None:
        tau *= min(1, args.tau_alpha_crit / args.alpha)

    dynamics = []
    for f, state, done in train_regular(f0, xtr, ytr, tau, args.train_time, args.alpha, partial(loss_func, args), bool(args.f0), args.max_dgrad, args.max_dout):
        with torch.no_grad():
            otr = f(xtr[j]) - otr0[j]
            ote = f(xte[j]) - ote0[j]

        if args.arch.split('_')[0] == 'fc':
            def getw(f, i):
                return torch.cat(list(getattr(f.f, "W{}".format(i))))
            state['wnorm'] = [getw(f, i).norm().item() for i in range(f.f.L + 1)]
            state['dwnorm'] = [(getw(f, i) - getw(f0, i)).norm().item() for i in range(f.f.L + 1)]

        state['train'] = {
            'loss': loss_func(args, otr * ytrj).mean().item(),
            'aloss': args.alpha * loss_func(args, otr * ytrj).mean().item(),
            'err': (otr * ytr[j] <= 0).double().mean().item(),
            'nd': (args.alpha * otr * ytr[j] < 1).long().sum().item(),
            'dfnorm': otr.pow(2).mean().sqrt(),
            'fnorm': (otr + otr0[j]).pow(2).mean().sqrt(),
            'outputs': otr if args.save_outputs else None,
            'labels': ytrj if args.save_outputs else None,
        }
        state['test'] = {
            'loss': loss_func(args, ote * ytej).mean().item(),
            'aloss': args.alpha * loss_func(args, ote * ytej).mean().item(),
            'err': (ote * yte[j] <= 0).double().mean().item(),
            'nd': (args.alpha * ote * yte[j] < 1).long().sum().item(),
            'dfnorm': ote.pow(2).mean().sqrt(),
            'fnorm': (ote + ote0[j]).pow(2).mean().sqrt(),
            'outputs': ote if args.save_outputs else None,
            'labels': ytej if args.save_outputs else None,
        }
        print("[i={d[step]:d} t={d[t]:.2e} wall={d[wall]:.0f}] [dt={d[dt]:.1e} dgrad={d[dgrad]:.1e} dout={d[dout]:.1e}] [train aL={d[train][aloss]:.2e} err={d[train][err]:.2f} nd={d[train][nd]}/{p}] [test aL={d[test][aloss]:.2e} err={d[test][err]:.2f}]".format(d=state, p=len(j)), flush=True)
        dynamics.append(state)

        if done or perf_counter() - t > 120:
            t = perf_counter()

            with torch.no_grad():
                otr = f(xtr) - otr0
                ote = f(xte) - ote0

            out = {
                'dynamics': dynamics,
                'train': {
                    'f0': otr0,
                    'outputs': otr,
                    'labels': ytr,
                },
                'test': {
                    'f0': ote0,
                    'outputs': ote,
                    'labels': yte,
                }
            }
            yield f, out


def run_exp(args, f0, xtr, ytr, xte, yte):
    run = {
        'args': args,
        'N': sum(p.numel() for p in f0.parameters()),
    }

    if args.delta_kernel == 1 or args.init_kernel == 1:
        init_kernel = compute_kernels(f0, xtr, xte[:len(xtr)])

    if args.init_kernel == 1:
        run['init_kernel'] = run_kernel(args, *init_kernel, f0, xtr, ytr, xte, yte)

    if args.delta_kernel == 1:
        init_kernel = (init_kernel[0].cpu(), init_kernel[2].cpu())
    elif args.init_kernel == 1:
        del init_kernel

    if args.regular == 1:
        for f, out in run_regular(args, f0, xtr, ytr, xte, yte):
            run['regular'] = out
            yield run

        if args.delta_kernel == 1 or args.final_kernel == 1:
            final_kernel = compute_kernels(f, xtr, xte[:len(xtr)])

        if args.final_kernel == 1:
            run['final_kernel'] = run_kernel(args, *final_kernel, f, xtr, ytr, xte, yte)

        if args.delta_kernel == 1:
            final_kernel = (final_kernel[0].cpu(), final_kernel[2].cpu())
            run['delta_kernel'] = {
                'train': (init_kernel[0] - final_kernel[0]).norm().item(),
                'test': (init_kernel[1] - final_kernel[1]).norm().item(),
            }

    yield run


def execute(args):
    torch.backends.cudnn.benchmark = True
    if args.dtype == 'float64':
        torch.set_default_dtype(torch.float64)
    if args.dtype == 'float32':
        torch.set_default_dtype(torch.float32)

    if args.d is None or args.d == 0:
        (xtr, ytr), (xte, yte) = get_binary_dataset(args.dataset, args.ptr, args.data_seed, args.device)
    else:
        (xtr, ytr), (xte, yte) = get_binary_pca_dataset(args.dataset, args.ptr, args.d, args.whitening, args.data_seed, args.device)

    xtr = xtr.type(torch.get_default_dtype())
    xte = xte.type(torch.get_default_dtype())
    ytr = ytr.type(torch.get_default_dtype())
    yte = yte.type(torch.get_default_dtype())

    assert len(xte) >= args.pte
    xte = xte[:args.pte]
    yte = yte[:args.pte]

    torch.manual_seed(args.init_seed + hash(args.alpha))

    arch, act = args.arch.split('_')
    if act == 'relu':
        act = lambda x: 2 ** 0.5 * torch.relu(x)
    elif act == 'tanh':
        act = torch.tanh
    elif act == 'softplus':
        factor = torch.nn.functional.softplus(torch.randn(100000, dtype=torch.float64), args.spbeta).pow(2).mean().rsqrt().item()
        act = lambda x: torch.nn.functional.softplus(x, beta=args.spbeta).mul(factor)
    else:
        raise ValueError('act not specified')

    if arch == 'fc':
        assert args.L is not None
        xtr = xtr.flatten(1)
        xte = xte.flatten(1)
        f = FC(xtr.size(1), args.h, args.L, act, args.bias).to(args.device)
    elif arch == 'cv':
        assert args.bias == 0
        f = CV(xtr.size(1), args.h, L1=args.cv_L1, L2=args.cv_L2, act=act, h_base=args.cv_h_base, fsz=args.cv_fsz, pad=args.cv_pad, stride_first=args.cv_stride_first).to(args.device)
    elif arch == 'resnet':
        assert args.bias == 0
        f = Wide_ResNet(xtr.size(1), 28, args.h, act, 1, args.mix_angle).to(args.device)
    else:
        raise ValueError('arch not specified')

    f = SplitEval(f, args.chunk)

    torch.manual_seed(args.batch_seed)
    for run in run_exp(args, f, xtr, ytr, xte, yte):
        yield run


def main():
    git = {
        'log': subprocess.getoutput('git log --format="%H" -n 1 -z'),
        'status': subprocess.getoutput('git status -z'),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--dtype", type=str, default='float64')

    parser.add_argument("--init_seed", type=int, required=True)
    parser.add_argument("--data_seed", type=int, required=True)
    parser.add_argument("--batch_seed", type=int, required=True)

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ptr", type=int, required=True)
    parser.add_argument("--pte", type=int)
    parser.add_argument("--d", type=int)
    parser.add_argument("--whitening", type=int, default=1)

    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--bias", type=float, default=0)
    parser.add_argument("--L", type=int)
    parser.add_argument("--h", type=int, required=True)
    parser.add_argument("--mix_angle", type=float, default=45)
    parser.add_argument("--spbeta", type=float, default=5.0)
    parser.add_argument("--cv_L1", type=int, default=2)
    parser.add_argument("--cv_L2", type=int, default=2)
    parser.add_argument("--cv_h_base", type=float, default=1)
    parser.add_argument("--cv_fsz", type=int, default=5)
    parser.add_argument("--cv_pad", type=int, default=1)
    parser.add_argument("--cv_stride_first", type=int, default=1)

    parser.add_argument("--init_kernel", type=int, required=True)
    parser.add_argument("--regular", type=int, default=1)
    parser.add_argument("--final_kernel", type=int, required=True)
    parser.add_argument("--store_kernel", type=int, default=0)
    parser.add_argument("--delta_kernel", type=int, default=0)
    parser.add_argument("--save_outputs", type=int, default=0)

    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--f0", type=int, default=1)

    parser.add_argument("--tau_over_h", type=float, default=0.0)
    parser.add_argument("--tau_alpha_crit", type=float)

    parser.add_argument("--train_time", type=float, required=True)
    parser.add_argument("--chunk", type=int)
    parser.add_argument("--max_dgrad", type=float, default=1e-4)
    parser.add_argument("--max_dout", type=float, default=1e-1)

    parser.add_argument("--loss", type=str, default="softhinge")
    parser.add_argument("--lossbeta", type=float, default=20.0)

    parser.add_argument("--pickle", type=str, required=True)
    args = parser.parse_args()

    if args.pte is None:
        args.pte = args.ptr

    if args.chunk is None:
        args.chunk = args.ptr

    torch.save(args, args.pickle)
    try:
        for res in execute(args):
            res['git'] = git
            with open(args.pickle, 'wb') as f:
                torch.save(args, f)
                torch.save(res, f)
    except:
        os.remove(args.pickle)
        raise


if __name__ == "__main__":
    main()
