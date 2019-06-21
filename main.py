# pylint: disable=C, R, E1101, bare-except
import argparse
import os
import subprocess

import torch

from archi import CV, FC, Wide_ResNet
from dataset import get_binary_dataset, get_binary_pca_dataset
from kernels import compute_kernels
from dynamics import train_kernel, train_regular


def run_kernel(args, ktrtr, ktetr, ktete, f, xtr, ytr, xte, yte):
    otr, dynamics = train_kernel(ktrtr, ytr, args.temp, args.tau, args.train_time, args.alpha, args.min_bs, args.max_bs, (args.df_min, args.df_max))
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

    def ev(f, x):
        return torch.cat([f(x[i: i + args.max_bs]) for i in range(0, len(x), args.max_bs)])

    with torch.no_grad():
        otr0 = ev(f0, xtr)
        ote0 = ev(f0, xte)

    def op(f, state):
        j = torch.randperm(min(len(xte), len(xtr)))[:10 * args.max_bs]
        with torch.no_grad():
            otr = ev(f, xtr[j]) - otr0[j]
            ote = ev(f, xte[j]) - ote0[j]

        if args.arch.split('_')[0] == 'fc':
            def getw(f, i):
                return torch.cat(list(getattr(f, "W{}".format(i))))
            state['wnorm'] = [getw(f, i).norm().item() for i in range(f.L + 1)]
            state['dwnorm'] = [(getw(f, i) - getw(f0, i)).norm().item() for i in range(f.L + 1)]

        state['train'] = {
            'loss': args.alpha ** -2 * (1 - args.alpha * otr * ytr[j]).relu().mean().item(),
            'err': (otr * ytr[j] <= 0).double().mean().item(),
            'nd': (args.alpha * otr * ytr[j] < 1).long().sum().item(),
            'dfnorm': otr.pow(2).mean().sqrt(),
            'fnorm': (otr + otr0[j]).pow(2).mean().sqrt(),
        }
        state['test'] = {
            'loss': args.alpha ** -2 * (1 - args.alpha * ote * yte[j]).relu().mean().item(),
            'err': (ote * yte[j] <= 0).double().mean().item(),
            'nd': (args.alpha * ote * yte[j] < 1).long().sum().item(),
            'dfnorm': ote.pow(2).mean().sqrt(),
            'fnorm': (ote + ote0[j]).pow(2).mean().sqrt(),
        }
        print("[i={d[step]:d} t={d[t]:.2e} wall={d[time]:.0f}] [dt={d[dt]:.1e} bs={d[bs]:d} df={d[df]:.1e}] [train L={d[train][loss]:.2e} err={d[train][err]:.2f} nd={d[train][nd]}/{p}] [test L={d[test][loss]:.2e} err={d[test][err]:.2f}]".format(d=state, p=len(j)), flush=True)

        return state

    f, dynamics = train_regular(f0, xtr, ytr, args.temp, args.tau, args.train_time, args.alpha, args.min_bs, args.max_bs, op, (args.df_min, args.df_max))

    with torch.no_grad():
        otr = ev(f, xtr) - otr0
        ote = ev(f, xte) - ote0

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
    return f, out


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
        f, out = run_regular(args, f0, xtr, ytr, xte, yte)
        run['regular'] = out

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

    return run


def execute(args):
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

    torch.manual_seed(args.init_seed + hash(args.alpha))

    arch, act = args.arch.split('_')
    if act == 'relu':
        act = lambda x: 2 ** 0.5 * torch.relu(x)
    elif act == 'tanh':
        act = torch.tanh
    elif act == 'softplus':
        act = lambda x: args.spsig * torch.nn.functional.softplus(x, beta=args.spbeta)
    else:
        raise ValueError('act not specified')

    if arch == 'fc':
        assert args.L is not None
        xtr = xtr.flatten(1)
        xte = xte.flatten(1)
        f = FC(xtr.size(1), args.h, args.L, act, beta=1).to(args.device)
    elif arch == 'cv':
        f = CV(xtr.size(1), args.h, h_base=1, L1=2, L2=2, act=act, fsz=5, beta=1, pad=1, stride_first=True).to(args.device)
    elif arch == 'resnet':
        f = Wide_ResNet(xtr.size(1), 28, args.h, act, 1, args.mix_angle).to(args.device)
    else:
        raise ValueError('arch not specified')

    torch.manual_seed(args.batch_seed)
    run = run_exp(args, f, xtr, ytr, xte, yte)

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
    parser.add_argument("--min_bs", type=int, default=1)
    parser.add_argument("--max_bs", type=int, required=True)

    parser.add_argument("--pickle", type=str, required=True)
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
