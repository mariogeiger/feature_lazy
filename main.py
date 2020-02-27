# pylint: disable=C, R, bare-except, arguments-differ, no-member, undefined-loop-variable
import argparse
import copy
import os
import subprocess
from functools import partial
from time import perf_counter

import torch

from archi import CV, FC, Wide_ResNet, FixedWeights, FixedAngles
from dataset import get_binary_dataset
from dynamics import train_kernel, train_regular
from kernels import compute_kernels
from mnas import MnasNetLike
from swish import SwishJit


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

    assert ktrtr.shape == (len(xtr), len(xtr))
    assert ktetr.shape == (len(xte), len(xtr))
    assert ktete.shape == (len(xte), len(xte))
    assert len(yte) == len(xte)
    assert len(ytr) == len(xtr)

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

        print(("[i={d[step]:d} t={d[t]:.2e} wall={d[wall]:.0f}] [dt={d[dt]:.1e} dgrad={d[dgrad]:.1e} dout={d[dout]:.1e}]"
              + " [train aL={d[train][aloss]:.2e} err={d[train][err]:.2f} nd={d[train][nd]}/{ptr}]").format(d=state, ptr=len(xtr)), flush=True)

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

    jtr = torch.randperm(len(xtr))[:10 * args.chunk].sort().values
    jte = torch.randperm(len(xte))[:10 * args.chunk].sort().values
    ytrj = ytr[jtr]
    ytej = yte[jte]

    tau = args.tau_over_h * args.h
    if args.tau_alpha_crit is not None:
        tau *= min(1, args.tau_alpha_crit / args.alpha)

    best_test_error = 1

    wall = perf_counter()
    dynamics = []
    for f, state, done in train_regular(f0, xtr, ytr, tau, args.alpha, partial(loss_func, args), bool(args.f0), args.chunk, args.max_dgrad, args.max_dout):
        with torch.no_grad():
            otr = f(xtr[jtr]) - otr0[jtr]
            ote = f(xte[jte]) - ote0[jte]

        state['wall'] = perf_counter() - wall
        state['norm'] = sum(p.norm().pow(2) for p in f.parameters()).sqrt().item()
        state['dnorm'] = sum((p0 - p).norm().pow(2) for p0, p in zip(f0.parameters(), f.parameters())).sqrt().item()

        if args.arch == 'fc':
            def getw(f, i):
                return torch.cat(list(getattr(f.f, "W{}".format(i))))
            state['wnorm'] = [getw(f, i).norm().item() for i in range(f.f.L + 1)]
            state['dwnorm'] = [(getw(f, i) - getw(f0, i)).norm().item() for i in range(f.f.L + 1)]

        test_err = (ote * ytej <= 0).double().mean().item()
        save_outputs = args.save_outputs
        if test_err < best_test_error:
            best_test_error = test_err
            save_outputs = True
            if not args.save_outputs:
                for x in dynamics:
                    x['train']['outputs'] = None
                    x['train']['labels'] = None
                    x['test']['outputs'] = None
                    x['test']['labels'] = None

        state['train'] = {
            'loss': loss_func(args, otr * ytrj).mean().item(),
            'aloss': args.alpha * loss_func(args, otr * ytrj).mean().item(),
            'err': (otr * ytrj <= 0).double().mean().item(),
            'nd': (args.alpha * otr * ytrj < 1).long().sum().item(),
            'dfnorm': otr.pow(2).mean().sqrt(),
            'fnorm': (otr + otr0[jtr]).pow(2).mean().sqrt(),
            'outputs': otr if save_outputs else None,
            'labels': ytrj if save_outputs else None,
        }
        state['test'] = {
            'loss': loss_func(args, ote * ytej).mean().item(),
            'aloss': args.alpha * loss_func(args, ote * ytej).mean().item(),
            'err': test_err,
            'nd': (args.alpha * ote * ytej < 1).long().sum().item(),
            'dfnorm': ote.pow(2).mean().sqrt(),
            'fnorm': (ote + ote0[jte]).pow(2).mean().sqrt(),
            'outputs': ote if save_outputs else None,
            'labels': ytej if save_outputs else None,
        }
        print(("[i={d[step]:d} t={d[t]:.2e} wall={d[wall]:.0f}] [dt={d[dt]:.1e} dgrad={d[dgrad]:.1e} dout={d[dout]:.1e}] "
              + "[train aL={d[train][aloss]:.2e} err={d[train][err]:.2f} nd={d[train][nd]}/{p}] [test aL={d[test][aloss]:.2e} "
              + "err={d[test][err]:.2f}]").format(d=state, p=len(ytrj)), flush=True)
        dynamics.append(state)

        if wall + args.train_time < perf_counter():
            done = True

        if done:
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
        else:
            out = {
                'dynamics': dynamics,
            }

        yield f, out, done
        if done:
            break


def run_exp(args, f0, xtr, ytr, xtk, ytk, xte, yte):
    run = {
        'args': args,
        'N': sum(p.numel() for p in f0.parameters()),
        'finished': False,
    }

    if args.delta_kernel == 1 or args.init_kernel == 1:
        init_kernel = compute_kernels(f0, xtk, xte[:len(xtk)])

    if args.init_kernel == 1:
        run['init_kernel'] = run_kernel(args, *init_kernel, f0, xtk, ytk, xte[:len(xtk)], yte[:len(xtk)])

    if args.init_kernel_ptr == 1:
        init_kernel_ptr = compute_kernels(f0, xtr, xte[:len(xtk)])
        run['init_kernel_ptr'] = run_kernel(args, *init_kernel_ptr, f0, xtr, ytr, xte[:len(xtk)], yte[:len(xtk)])
        del init_kernel_ptr

    if args.delta_kernel == 1:
        init_kernel = (init_kernel[0].cpu(), init_kernel[2].cpu())
    elif args.init_kernel == 1:
        del init_kernel

    if args.regular == 1:
        if args.running_kernel:
            it = iter(args.running_kernel)
            al = next(it)
        else:
            al = -1
        t = perf_counter()
        for f, out, done in run_regular(args, f0, xtr, ytr, xte, yte):
            run['regular'] = out
            if out['dynamics'][-1]['train']['aloss'] < al * out['dynamics'][0]['train']['aloss']:
                try:
                    al = next(it)
                except StopIteration:
                    al = 0

                running_kernel = compute_kernels(f, xtk, xte[:len(xtk)])
                out['dynamics'][-1]['kernel'] = run_kernel(args, *running_kernel, f, xtk, ytk, xte[:len(xtk)], yte[:len(xtk)])
                if args.ptr < args.ptk:
                    ktktk, ktetk, ktete = running_kernel
                    ktktk = ktktk[:len(xtr)][:, :len(xtr)]
                    ktetk = ktetk[:, :len(xtr)]
                    out['dynamics'][-1]['kernel_ptr'] = run_kernel(args, ktktk, ktetk, ktete, f, xtk[:len(xtr)], ytk[:len(xtr)], xte[:len(xtk)], yte[:len(xtk)])
                else:
                    out['dynamics'][-1]['kernel_ptr'] = out['dynamics'][-1]['kernel']
                out['dynamics'][-1]['state'] = copy.deepcopy(f.state_dict())

            if done or perf_counter() - t > 120:
                t = perf_counter()
                yield run

        if args.delta_kernel == 1 or args.final_kernel == 1:
            final_kernel = compute_kernels(f, xtk, xte[:len(xtk)])
            if args.final_kernel_ptr == 1:
                ktktk, ktetk, ktete = final_kernel
                ktktk = ktktk[:len(xtr)][:, :len(xtr)]
                ktetk = ktetk[:, :len(xtr)]
                final_kernel_ptr = (ktktk, ktetk, ktete)

        elif args.final_kernel_ptr == 1:
            final_kernel_ptr = compute_kernels(f, xtk[:len(xtr)], xte[:len(xtk)])

        if args.final_kernel == 1:
            run['final_kernel'] = run_kernel(args, *final_kernel, f, xtk, ytk, xte[:len(xtk)], yte[:len(xtk)])

        if args.final_kernel_ptr == 1:
            assert len(xtk) >= len(xtr)
            run['final_kernel_ptr'] = run_kernel(args, *final_kernel_ptr, f, xtk[:len(xtr)], ytk[:len(xtr)], xte[:len(xtk)], yte[:len(xtk)])

        if args.delta_kernel == 1:
            final_kernel = (final_kernel[0].cpu(), final_kernel[2].cpu())
            run['delta_kernel'] = {
                'train': (init_kernel[0] - final_kernel[0]).norm().item(),
                'test': (init_kernel[1] - final_kernel[1]).norm().item(),
            }

    run['finished'] = True
    yield run


def init(args):
    torch.backends.cudnn.benchmark = True
    if args.dtype == 'float64':
        torch.set_default_dtype(torch.float64)
    if args.dtype == 'float32':
        torch.set_default_dtype(torch.float32)

    # if args.d is None or args.d == 0:
    x, y = get_binary_dataset(args.dataset, args.ptr + args.ptk + args.pte, args.d, args.data_seed, args.device)
    # else:
    #     (xtr, ytr), (xte, yte) = get_binary_pca_dataset(args.dataset, args.ptr + args.ptk, args.d, args.whitening, args.data_seed, args.device)

    xtr = x[:args.ptr]
    ytr = y[:args.ptr]
    xtk = x[args.ptr: args.ptr + args.ptk]
    ytk = y[args.ptr: args.ptr + args.ptk]
    xte = x[args.ptr + args.ptk:]
    yte = y[args.ptr + args.ptk:]

    xtr = xtr.type(torch.get_default_dtype())
    xtk = xtk.type(torch.get_default_dtype())
    xte = xte.type(torch.get_default_dtype())
    ytr = ytr.type(torch.get_default_dtype())
    yte = yte.type(torch.get_default_dtype())
    ytk = ytk.type(torch.get_default_dtype())

    assert len(xte) >= args.pte
    xte = xte[:args.pte]
    yte = yte[:args.pte]

    torch.manual_seed(args.init_seed + hash(args.alpha))

    if args.act == 'relu':
        def act(x):
            return 2 ** 0.5 * torch.relu(x)
    elif args.act == 'tanh':
        act = torch.tanh
    elif args.act == 'softplus':
        factor = torch.nn.functional.softplus(torch.randn(100000, dtype=torch.float64), args.spbeta).pow(2).mean().rsqrt().item()

        def act(x):
            return torch.nn.functional.softplus(x, beta=args.spbeta).mul(factor)
    elif args.act == 'swish':
        act = SwishJit()
    else:
        raise ValueError('act not specified')

    if args.arch == 'fc':
        assert args.L is not None
        xtr = xtr.flatten(1)
        xtk = xtk.flatten(1)
        xte = xte.flatten(1)
        f = FC(xtr.size(1), args.h, args.L, act, args.bias)
    elif args.arch == 'cv':
        assert args.bias == 0
        f = CV(xtr.size(1), args.h, L1=args.cv_L1, L2=args.cv_L2, act=act, h_base=args.cv_h_base,
               fsz=args.cv_fsz, pad=args.cv_pad, stride_first=args.cv_stride_first)
    elif args.arch == 'resnet':
        assert args.bias == 0
        f = Wide_ResNet(xtr.size(1), 28, args.h, act, 1, args.mix_angle)
    elif args.arch == 'mnas':
        assert args.act == 'swish'
        f = MnasNetLike(xtr.size(1), args.h, args.cv_L1, args.cv_L2, dim=xtr.dim() - 2)
    elif args.arch == 'fixed_weights':
        f = FixedWeights(args.d, args.h, act, args.bias)
    elif args.arch == 'fixed_angles':
        f = FixedAngles(args.d, args.h, act, args.bias)
    else:
        raise ValueError('arch not specified')

    f = SplitEval(f, args.chunk)
    f = f.to(args.device)

    return f, xtr, ytr, xtk, ytk, xte, yte


def execute(args):
    f, xtr, ytr, xtk, ytk, xte, yte = init(args)

    torch.manual_seed(args.batch_seed)
    for run in run_exp(args, f, xtr, ytr, xtk, ytk, xte, yte):
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
    parser.add_argument("--ptk", type=int, default=0)
    parser.add_argument("--pte", type=int)
    parser.add_argument("--d", type=int)
    parser.add_argument("--whitening", type=int, default=1)

    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--act", type=str, required=True)
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

    parser.add_argument("--init_kernel", type=int, default=0)
    parser.add_argument("--init_kernel_ptr", type=int, default=0)
    parser.add_argument("--regular", type=int, default=1)
    parser.add_argument('--running_kernel', nargs='+', type=float)
    parser.add_argument("--final_kernel", type=int, default=0)
    parser.add_argument("--final_kernel_ptr", type=int, default=0)
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
