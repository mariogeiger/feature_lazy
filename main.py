# pylint: disable=C, R, E1101, bare-except
import argparse
import os
import subprocess

import torch

import time_logging
from archi import CV, FC, Wide_ResNet, normal_orthogonal_
from dataset import get_binary_dataset, get_binary_pca_dataset
from kernels import compute_kernels, kernel_likelihood
from dynamics import train_kernel, train_regular


def run_kernel(args, f, xtr, ytr, xte, yte):
    ktrtr, ktetr, ktete = compute_kernels(f, xtr, xte[:len(xtr)])

    otr, dynamics = train_kernel(ktrtr, ytr, args.temp, args.tau, args.train_time, args.alpha, (args.df_min, args.df_max))
    c = torch.gels(otr.view(-1, 1), ktrtr)[0].flatten()

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
                'nll': kernel_likelihood(ktrtr, ytr),
                'diag': ktrtr.diag(),
                'mean': ktrtr.mean(),
                'std': ktrtr.std(),
                'norm': ktrtr.norm(),
            },
            'test': {
                'value': ktete.cpu() if args.store_kernel == 1 else None,
                'nll': kernel_likelihood(ktete, yte[:len(xtr)]),
                'diag': ktete.diag(),
                'mean': ktete.mean(),
                'std': ktete.std(),
                'norm': ktete.norm(),
            },
        },
    }

    return out, (ktrtr.cpu(), ktete.cpu()) if args.delta_kernel == 1 else None


def run_regular(args, f0, xtr, ytr, xte, yte):

    def op(f, state):
        j = torch.randperm(min(len(xte), len(xtr)))[:10 * args.chunk]
        with torch.no_grad():
            otr0 = torch.cat([f0(xtr[j[i: i + args.chunk]]) for i in range(0, len(j), args.chunk)])
            ote0 = torch.cat([f0(xte[j[i: i + args.chunk]]) for i in range(0, len(j), args.chunk)])
            otr  = torch.cat([ f(xtr[j[i: i + args.chunk]]) for i in range(0, len(j), args.chunk)]) - otr0
            ote  = torch.cat([ f(xte[j[i: i + args.chunk]]) for i in range(0, len(j), args.chunk)]) - ote0

        last_w = "W{}".format(args.L)
        if hasattr(f, last_w):
            state['last_norm'] = getattr(f, last_w).norm().item()
            state['last_dnorm'] = (getattr(f0, last_w) - getattr(f, last_w)).norm().item()

        state['train'] = {
            'loss': args.alpha ** -2 * (1 - args.alpha * otr * ytr[j]).relu().mean().item(),
            'err': (otr * ytr[j] <= 0).double().mean().item(),
            'nd': (args.alpha * otr * ytr[j] < 1).long().sum().item(),
            'dfnorm': otr.pow(2).mean().sqrt(),
            'fnorm': (otr + otr0).pow(2).mean().sqrt(),
        }
        state['test'] = {
            'loss': args.alpha ** -2 * (1 - args.alpha * ote * yte[j]).relu().mean().item(),
            'err': (ote * yte[j] <= 0).double().mean().item(),
            'nd': (args.alpha * ote * yte[j] < 1).long().sum().item(),
            'dfnorm': ote.pow(2).mean().sqrt(),
            'fnorm': (ote + ote0).pow(2).mean().sqrt(),
        }
        print("[i={d[step]:d} t={d[t]:.2e} wall={d[time]:.0f}] [dt={d[dt]:.1e} bs={d[bs]:d} df={d[df]:.1e}] [train L={d[train][loss]:.2e} err={d[train][err]:.2f} nd={d[train][nd]}] [test L={d[test][loss]:.2e} err={d[test][err]:.2f}]".format(d=state), flush=True)

        return state

    f, dynamics = train_regular(f0, xtr, ytr, args.temp, args.tau, args.train_time, args.alpha, args.chunk, op, (args.df_min, args.df_max))
    with torch.no_grad():
        otr = torch.cat([f(xtr[i: i + args.chunk]) - f0(xtr[i: i + args.chunk]) for i in range(0, len(xtr), args.chunk)])
        ote = torch.cat([f(xte[i: i + args.chunk]) - f0(xte[i: i + args.chunk]) for i in range(0, len(xte), args.chunk)])

    out = {
        'dynamics': dynamics,
        'train': {
            'outputs': otr,
            'labels': ytr,
        },
        'test': {
            'outputs': ote,
            'labels': yte,
        }
    }
    return f, out


def run_exp(args, f, xtr, ytr, xte, yte):
    time = time_logging.start()
    run = {
        'args': args,
        'N': sum(p.numel() for p in f.parameters()),
    }

    if args.init_kernel == 1:
        run['init_kernel'], init_kernel = run_kernel(args, f, xtr, ytr, xte, yte)
        time = time_logging.end("init_kernel", time)

    if args.regular == 1:
        f, out = run_regular(args, f, xtr, ytr, xte, yte)
        run['regular'] = out
        time = time_logging.end("regular", time)


        if args.final_kernel == 1:
            run['final_kernel'], final_kernel = run_kernel(args, f, xtr, ytr, xte, yte)
            time = time_logging.end("final_kernel", time)

    if args.delta_kernel == 1:
        assert args.init_kernel == 1
        assert args.final_kernel == 1
        run['delta_kernel'] = {
            'train': (init_kernel[0] - final_kernel[0]).norm().item(),
            'test': (init_kernel[1] - final_kernel[1]).norm().item(),
        }

    return run


def execute(args):
    time = time_logging.start()
    torch.backends.cudnn.benchmark = True
    if args.dtype == 'float64':
        torch.set_default_dtype(torch.float64)
    if args.dtype == 'float32':
        torch.set_default_dtype(torch.float32)

    if args.d is None:
        (xtr, ytr), (xte, yte) = get_binary_dataset(args.dataset, args.ptr, args.data_seed, args.device)
    else:
        (xtr, ytr), (xte, yte) = get_binary_pca_dataset(args.dataset, args.ptr, args.d, args.data_seed, args.device)

    xtr = xtr.type(torch.get_default_dtype())
    xte = xte.type(torch.get_default_dtype())
    ytr = ytr.type(torch.get_default_dtype())
    yte = yte.type(torch.get_default_dtype())

    assert len(xte) >= args.pte
    xte = xte[:args.pte]
    yte = yte[:args.pte]

    if args.init == 'normal':
        init_ = torch.nn.init.normal_
    if args.init == 'orth':
        init_ = normal_orthogonal_

    torch.manual_seed(args.init_seed + hash(args.alpha))
    if args.arch == 'cv':
        assert args.init == 'normal'
        f = CV(xtr.size(1), args.h, h_base=1, L1=2, L2=2, act=lambda x: 2 ** 0.5 * torch.relu(x), fsz=5, beta=1, pad=1, stride_first=True, split_w=True).to(args.device)
    if args.arch == 'wide_resnet':
        f = Wide_ResNet(xtr.size(1), 28, args.h, 1, args.mix_angle, init_).to(args.device)
    if args.arch == 'fc_relu':
        assert args.init == 'normal'
        assert args.L is not None
        f = FC(args.d, args.h, args.L, lambda x: 2 ** 0.5 * torch.relu(x), 1).to(args.device)
    if args.arch == 'fc_tanh':
        assert args.init == 'normal'
        assert args.L is not None
        f = FC(args.d, args.h, args.L, torch.tanh, 1).to(args.device)
    if args.arch == 'fc_softplus':
        assert args.init == 'normal'
        assert args.L is not None
        f = FC(args.d, args.h, args.L, lambda x: args.spsig * torch.nn.functional.softplus(x, beta=args.spbeta), 1).to(args.device)

    torch.manual_seed(args.batch_seed)
    run = run_exp(args, f, xtr, ytr, xte, yte)

    time = time_logging.end("total", time)
    run['time_statistics'] = time_logging.text_statistics()

    return run


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

    parser.add_argument("--df_min", type=float, default=1e-4)
    parser.add_argument("--df_max", type=float, default=1e-2)

    parser.add_argument("--dataset", type=str)
    parser.add_argument("--ptr", type=int, required=True)
    parser.add_argument("--pte", type=int)
    parser.add_argument("--d", type=int)

    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--L", type=int)
    parser.add_argument("--h", type=int, required=True)
    parser.add_argument("--mix_angle", type=float, default=45)
    parser.add_argument("--spbeta", type=float, default=1.0)
    parser.add_argument("--spsig", type=float, default=1.0)
    parser.add_argument("--init", type=str, default='normal')

    parser.add_argument("--init_kernel", type=int, required=True)
    parser.add_argument("--regular", type=int, default=1)
    parser.add_argument("--final_kernel", type=int, required=True)
    parser.add_argument("--store_kernel", type=int, default=0)
    parser.add_argument("--delta_kernel", type=int, default=0)

    parser.add_argument("--alpha", type=float)
    parser.add_argument("--sqhalpha", type=float)

    parser.add_argument("--temp", type=float)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--train_time", type=float, required=True)
    parser.add_argument("--chunk", type=int, required=True)

    parser.add_argument("--pickle", type=str, required=True)
    parser.add_argument("--desc", type=str, required=True)
    args = parser.parse_args()

    if args.pte is None:
        args.pte = args.ptr

    if args.alpha is None and args.sqhalpha is None:
        args.alpha = 1

    if args.sqhalpha is None:
        args.sqhalpha = args.h ** 0.5 * args.alpha

    if args.alpha is None:
        args.alpha = args.sqhalpha / args.h ** 0.5

    torch.save(args, args.pickle)
    try:
        res = execute(args)
        res['git'] = git
        with open(args.pickle, 'wb') as f:
            torch.save(args, f)
            torch.save(res, f)
    except:
        os.remove(args.pickle)
        raise


if __name__ == "__main__":
    main()
