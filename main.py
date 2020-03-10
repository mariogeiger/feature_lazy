# pylint: disable=C, R, bare-except, arguments-differ, no-member, undefined-loop-variable
import argparse
import copy
import os
import subprocess
from functools import partial
from time import perf_counter

import torch

from arch import CV, FC, FixedAngles, FixedWeights, Wide_ResNet
from arch.mnas import MnasNetLike
from arch.swish import swish
from dataset import get_binary_dataset
from dynamics import train_kernel, train_regular, loglinspace
from kernels import compute_kernels


def loss_func(args, f, y):
    if args.loss == 'softhinge':
        sp = partial(torch.nn.functional.softplus, beta=args.loss_beta)
        return sp(args.loss_margin - args.alpha * f * y) / args.alpha
    if args.loss == 'qhinge':
        return 0.5 * (args.loss_margin - args.alpha * f * y).relu().pow(2) / args.alpha


def loss_func_prime(args, f, y):
    if args.loss == 'softhinge':
        return -torch.sigmoid(args.loss_beta * (args.loss_margin - args.alpha * f * y)) * y
    if args.loss == 'qhinge':
        return -(args.loss_margin - args.alpha * f * y).relu() * y


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

    tau = args.tau_over_h * args.h
    if args.tau_alpha_crit is not None:
        tau *= min(1, args.tau_alpha_crit / args.alpha)

    margin = 0

    checkpoint_generator = loglinspace(100, 100 * 100)
    checkpoint = next(checkpoint_generator)

    wall = perf_counter()
    dynamics = []
    for state, otr, _velo, grad in train_kernel(ktrtr, ytr, tau, args.alpha, partial(loss_func_prime, args), args.max_dgrad, args.max_dout):
        save = False
        save_outputs = False
        stop = False

        if state['step'] == checkpoint:
            checkpoint = next(checkpoint_generator)
            save = True
        if (args.alpha * otr * ytr).min() > margin:
            margin += 0.5
            save = True
            save_outputs = True
        if torch.isnan(otr).any():
            save = True
            stop = True

        if not save:
            continue

        if args.save_outputs:
            save_outputs = True

        if (args.alpha * otr * ytr).min() > args.stop_margin:
            save_outputs = True
            stop = True

        if wall + args.train_time < perf_counter():
            save_outputs = True
            stop = True

        state['grad_norm'] = grad.norm().item()
        state['wall'] = perf_counter() - wall

        state['train'] = {
            'loss': loss_func(args, otr, ytr).mean().item(),
            'aloss': args.alpha * loss_func(args, otr, ytr).mean().item(),
            'err': (otr * ytr <= 0).double().mean().item(),
            'nd': (args.alpha * otr * ytr < args.stop_margin).long().sum().item(),
            'mind': (args.alpha * otr * ytr).min().item(),
            'dfnorm': otr.pow(2).mean().sqrt(),
            'outputs': otr if save_outputs else None,
            'labels': ytr if save_outputs else None,
        }
        state['test'] = None

        if save_outputs:
            c = torch.lstsq(otr.view(-1, 1), ktrtr).solution.flatten()

            if len(xte) > len(xtr):
                from hessian import gradient
                a = gradient(f(xtr) @ c, f.parameters())
                ote = torch.stack([gradient(f(x[None]), f.parameters()) @ a for x in xte])
            else:
                ote = ktetr @ c

            state['test'] = {
                'loss': loss_func(args, ote, yte).mean().item(),
                'aloss': args.alpha * loss_func(args, ote, yte).mean().item(),
                'err': (ote * yte <= 0).double().mean().item(),
                'nd': (args.alpha * ote * yte < args.stop_margin).long().sum().item(),
                'mind': (args.alpha * ote * yte).min().item(),
                'dfnorm': ote.pow(2).mean().sqrt(),
                'outputs': ote,
                'labels': yte,
            }

        print(("[i={d[step]:d} t={d[t]:.2e} wall={d[wall]:.0f}] [dt={d[dt]:.1e} dgrad={d[dgrad]:.1e} dout={d[dout]:.1e}]"
              + " [train aL={d[train][aloss]:.2e} err={d[train][err]:.2f} nd={d[train][nd]}/{ptr}]").format(d=state, ptr=len(xtr)), flush=True)
        dynamics.append(state)

        out = {
            'dynamics': dynamics,
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

        yield out
        if stop:
            break


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
    margin = 0

    checkpoint_generator = loglinspace(100, 100 * 100)
    checkpoint = next(checkpoint_generator)

    wall = perf_counter()
    dynamics = []
    for state, f, otr, otr0, grad in train_regular(f0, xtr, ytr, tau, args.alpha, partial(loss_func, args), bool(args.f0), args.chunk, args.max_dgrad, args.max_dout):
        otr = otr - otr0

        save = False
        save_outputs = False
        stop = False

        if state['step'] == checkpoint:
            checkpoint = next(checkpoint_generator)
            save = True
        if (args.alpha * otr * ytr).min() > margin:
            margin += 0.5
            save = True
            save_outputs = True
        if torch.isnan(otr).any():
            save = True
            stop = True

        if not save:
            continue

        with torch.no_grad():
            ote = f(xte) - ote0

        if args.save_outputs:
            save_outputs = True

        if (args.alpha * otr * ytr).min() > args.stop_margin:
            save_outputs = True
            stop = True

        if wall + args.train_time < perf_counter():
            save_outputs = True
            stop = True

        test_err = (ote * yte <= 0).double().mean().item()
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
            'err': (otr * ytr <= 0).double().mean().item(),
            'nd': (args.alpha * otr * ytr < args.stop_margin).long().sum().item(),
            'mind': (args.alpha * otr * ytr).min().item(),
            'dfnorm': otr.pow(2).mean().sqrt(),
            'fnorm': (otr + otr0).pow(2).mean().sqrt(),
            'outputs': otr if save_outputs else None,
            'labels': ytr if save_outputs else None,
        }
        state['test'] = {
            'loss': loss_func(args, ote, yte).mean().item(),
            'aloss': args.alpha * loss_func(args, ote, yte).mean().item(),
            'err': test_err,
            'nd': (args.alpha * ote * yte < args.stop_margin).long().sum().item(),
            'mind': (args.alpha * ote * yte).min().item(),
            'dfnorm': ote.pow(2).mean().sqrt(),
            'fnorm': (ote + ote0).pow(2).mean().sqrt(),
            'outputs': ote if save_outputs else None,
            'labels': yte if save_outputs else None,
        }
        print(("[i={d[step]:d} t={d[t]:.2e} wall={d[wall]:.0f}] [dt={d[dt]:.1e} dgrad={d[dgrad]:.1e} dout={d[dout]:.1e}] "
              + "[train aL={d[train][aloss]:.2e} err={d[train][err]:.2f} nd={d[train][nd]}/{p} mind={d[train][mind]:.3f}] "
              + "[test aL={d[test][aloss]:.2e} err={d[test][err]:.2f}]").format(d=state, p=len(ytr)), flush=True)
        dynamics.append(state)

        out = {
            'dynamics': dynamics,
        }

        yield f, out
        if stop:
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
        for out in run_kernel(args, *init_kernel, f0, xtk, ytk, xte[:len(xtk)], yte[:len(xtk)]):
            run['init_kernel'] = out

    if args.init_kernel_ptr == 1:
        init_kernel_ptr = compute_kernels(f0, xtr, xte[:len(xtk)])
        for out in run_kernel(args, *init_kernel_ptr, f0, xtr, ytr, xte[:len(xtk)], yte[:len(xtk)]):
            run['init_kernel_ptr'] = out
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
        wall = perf_counter()
        for f, out in run_regular(args, f0, xtr, ytr, xte, yte):
            run['regular'] = out
            if out['dynamics'][-1]['train']['aloss'] < al * out['dynamics'][0]['train']['aloss']:
                try:
                    al = next(it)
                except StopIteration:
                    al = 0

                running_kernel = compute_kernels(f, xtk, xte[:len(xtk)])
                for kout in run_kernel(args, *running_kernel, f, xtk, ytk, xte[:len(xtk)], yte[:len(xtk)]):
                    out['dynamics'][-1]['kernel'] = kout
                if args.ptr < args.ptk:
                    ktktk, ktetk, ktete = running_kernel
                    ktktk = ktktk[:len(xtr)][:, :len(xtr)]
                    ktetk = ktetk[:, :len(xtr)]
                    for kout in run_kernel(args, ktktk, ktetk, ktete, f, xtk[:len(xtr)], ytk[:len(xtr)], xte[:len(xtk)], yte[:len(xtk)]):
                        out['dynamics'][-1]['kernel_ptr'] = kout
                else:
                    out['dynamics'][-1]['kernel_ptr'] = out['dynamics'][-1]['kernel']
                out['dynamics'][-1]['state'] = copy.deepcopy(f.state_dict())

            if perf_counter() - wall > 120:
                wall = perf_counter()
                yield run
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
            for out in run_kernel(args, *final_kernel, f, xtk, ytk, xte[:len(xtk)], yte[:len(xtk)]):
                run['final_kernel'] = out

        if args.final_kernel_ptr == 1:
            assert len(xtk) >= len(xtr)
            for out in run_kernel(args, *final_kernel_ptr, f, xtk[:len(xtr)], ytk[:len(xtr)], xte[:len(xtk)], yte[:len(xtk)]):
                run['final_kernel_ptr'] = out

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

    [(xte, yte, ite), (xtk, ytk, itk), (xtr, ytr, itr)] = get_binary_dataset(
        args.dataset,
        (args.pte, args.ptk, args.ptr),
        (args.seed_testset + args.pte, args.seed_kernelset + args.ptk, args.seed_trainset + args.ptr),
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

    if args.arch == 'fc':
        assert args.L is not None
        xtr = xtr.flatten(1)
        xtk = xtk.flatten(1)
        xte = xte.flatten(1)
        f = FC(xtr.size(1), args.h, 1, args.L, act, args.bias)
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

    return f, xtr, ytr, itr, xtk, ytk, itk, xte, yte, ite


def execute(args):
    f, xtr, ytr, itr, xtk, ytk, itk, xte, yte, ite = init(args)

    torch.manual_seed(0)
    for run in run_exp(args, f, xtr, ytr, xtk, ytk, xte, yte):
        run['dataset'] = {
            'test': ite,
            'kernel': itk,
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
    parser.add_argument("--seed_testset", type=int, default=0, help="determines the testset, will affect the kernelset and trainset as well")
    parser.add_argument("--seed_kernelset", type=int, default=0, help="determines the kernelset, will affect the trainset as well")
    parser.add_argument("--seed_trainset", type=int, default=0, help="determines the trainset")

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ptr", type=int, required=True)
    parser.add_argument("--ptk", type=int, default=0)
    parser.add_argument("--pte", type=int)
    parser.add_argument("--d", type=int)
    parser.add_argument("--whitening", type=int, default=1)

    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--act", type=str, required=True)
    parser.add_argument("--act_beta", type=float, default=5.0)
    parser.add_argument("--bias", type=float, default=0)
    parser.add_argument("--L", type=int)
    parser.add_argument("--h", type=int, required=True)
    parser.add_argument("--mix_angle", type=float, default=45)
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
    parser.add_argument("--save_state", type=int, default=0)

    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--f0", type=int, default=1)

    parser.add_argument("--tau_over_h", type=float, default=0.0)
    parser.add_argument("--tau_alpha_crit", type=float)

    parser.add_argument("--train_time", type=float, required=True)
    parser.add_argument("--chunk", type=int)
    parser.add_argument("--max_dgrad", type=float, default=1e-4)
    parser.add_argument("--max_dout", type=float, default=1e-1)

    parser.add_argument("--loss", type=str, default="softhinge")
    parser.add_argument("--loss_beta", type=float, default=20.0)
    parser.add_argument("--loss_margin", type=float, default=1.0)
    parser.add_argument("--stop_margin", type=float, default=1.0)

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
