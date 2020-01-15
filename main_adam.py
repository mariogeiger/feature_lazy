# pylint: disable=bare-except, arguments-differ, no-member, missing-docstring, invalid-name, line-too-long
import argparse
import copy
import itertools
import os
import subprocess
from time import perf_counter

import torch

from archi import CV, FC, Wide_ResNet
from dataset import get_binary_dataset, get_binary_pca_dataset
from dynamics import loglinspace


class SplitEval(torch.nn.Module):
    def __init__(self, f, size):
        super().__init__()
        self.f = f
        self.size = size

    def forward(self, x):
        return torch.cat([self.f(x[i: i + self.size]) for i in range(0, len(x), self.size)])


def hinge(out, y, alpha):
    return (1 - alpha * out * y).relu().mean() / alpha


def quad_hinge(out, y, alpha):
    return 0.5 * (1 - alpha * out * y).relu().pow(2).mean() / alpha ** 2


def mse(out, y, alpha):
    return 0.5 * (1.1 - alpha * out * y).pow(2).mean() / alpha ** 2


def run_regular(args, f0, loss, xtr, ytr, xte, yte):

    f0.train(True)
    with torch.no_grad():
        otr0 = f0(xtr)
        ote0 = f0(xte)

    f = copy.deepcopy(f0)
    optimizer = torch.optim.Adam(f.parameters(), args.lr)

    dynamics = []
    checkpoint_generator = loglinspace(0.1, 1000)
    checkpoint = next(checkpoint_generator)
    wall = perf_counter()

    for step in itertools.count():

        batch = torch.randperm(len(xtr))[:args.bs]
        xb = xtr[batch]

        f.train(True)
        loss_value = loss(f(xb) - otr0[batch], ytr[batch], args.alpha)

        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        save = False

        if step == checkpoint:
            checkpoint = next(checkpoint_generator)
            assert checkpoint > step
            save = True

        if save:
            assert len(xtr) < len(xte)
            j = torch.randperm(len(xtr))

            f.train(True)
            with torch.no_grad():
                otr = f(xtr[j]) - otr0[j]
                ote = f(xte[j]) - ote0[j]

            state = {
                'step': step,
                'wall': perf_counter() - wall,
                'batch_loss': loss_value.item(),
                'norm': sum(p.norm().pow(2) for p in f.parameters()).sqrt().item(),
                'dnorm': sum((p0 - p).norm().pow(2) for p0, p in zip(f0.parameters(), f.parameters())).sqrt().item(),
                'train': {
                    'loss': loss(otr, ytr[j], args.alpha).item(),
                    'aloss': args.alpha * loss(otr, ytr[j], args.alpha).item(),
                    'aaloss': args.alpha ** 2 * loss(otr, ytr[j], args.alpha).item(),
                    'err': (otr * ytr[j] <= 0).double().mean().item(),
                    'nd': (args.alpha * otr * ytr[j] < 1).long().sum().item(),
                    'dfnorm': otr.pow(2).mean().sqrt(),
                    'fnorm': (otr + otr0[j]).pow(2).mean().sqrt(),
                },
                'test': {
                    'loss': loss(ote, yte[j], args.alpha).item(),
                    'aloss': args.alpha * loss(ote, yte[j], args.alpha).item(),
                    'aaloss': args.alpha ** 2 * loss(ote, yte[j], args.alpha).item(),
                    'err': (ote * yte[j] <= 0).double().mean().item(),
                    'nd': (args.alpha * ote * yte[j] < 1).long().sum().item(),
                    'dfnorm': ote.pow(2).mean().sqrt(),
                    'fnorm': (ote + ote0[j]).pow(2).mean().sqrt(),
                },
            }

            if args.arch.split('_')[0] == 'fc':
                def getw(f, i):
                    return torch.cat(list(getattr(f.f, "W{}".format(i))))
                state['wnorm'] = [getw(f, i).norm().item() for i in range(f.f.L + 1)]
                state['dwnorm'] = [(getw(f, i) - getw(f0, i)).norm().item() for i in range(f.f.L + 1)]

            print("[i={d[step]:d} wall={d[wall]:.0f}] [train aL={d[train][aloss]:.2e} err={d[train][err]:.2f} nd={d[train][nd]}/{p}] [test aL={d[test][aloss]:.2e} err={d[test][err]:.2f}]".format(d=state, p=len(j)), flush=True)

            dynamics.append(state)

            if state['train']['nd'] == 0:
                break

        if perf_counter() > wall + args.train_time:
            break

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
    return f, out


def run_exp(args, f0, xtr, ytr, xte, yte):
    run = {
        'args': args,
        'N': sum(p.numel() for p in f0.parameters()),
    }

    if args.loss == 'hinge':
        loss = hinge
    if args.loss == 'quad_hinge':
        loss = quad_hinge
    if args.loss == 'mse':
        loss = mse

    _f, out = run_regular(args, f0, loss, xtr, ytr, xte, yte)
    run['regular'] = out

    return run


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

    if '_' in args.arch:
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
            f = FC(xtr.size(1), args.h, args.L, act).to(args.device)
        elif arch == 'cv':
            f = CV(xtr.size(1), args.h, h_base=1, L1=2, L2=2, act=act, fsz=5, pad=1, stride_first=True).to(args.device)
        elif arch == 'resnet':
            f = Wide_ResNet(xtr.size(1), 28, args.h, act, 1, args.mix_angle).to(args.device)
        else:
            raise ValueError('arch not specified')
    else:
        archi, repo = args.arch.split('@')
        f = torch.hub.load(repo, archi, pretrained=False, num_classes=1)
        f = f.to(dtype=torch.get_default_dtype(), device=args.device)

    f = SplitEval(f, args.chunk)

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

    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--bs", type=int, required=True)

    parser.add_argument("--dataset", type=str)
    parser.add_argument("--ptr", type=int, required=True)
    parser.add_argument("--pte", type=int)
    parser.add_argument("--d", type=int)
    parser.add_argument("--whitening", type=int, default=1)

    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--L", type=int)
    parser.add_argument("--h", type=int, required=True)
    parser.add_argument("--mix_angle", type=float, default=45)
    parser.add_argument("--spbeta", type=float, default=5.0)

    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--loss", type=str, required=True)

    parser.add_argument("--train_time", type=float, required=True)
    parser.add_argument("--chunk", type=int)

    parser.add_argument("--pickle", type=str, required=True)
    args = parser.parse_args()

    if args.pte is None:
        args.pte = args.ptr

    if args.chunk is None:
        args.chunk = args.ptr

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
