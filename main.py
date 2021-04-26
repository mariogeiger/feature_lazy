# pylint: disable=C, R, bare-except, arguments-differ, no-member, undefined-loop-variable, not-callable, unbalanced-tuple-unpacking, abstract-method
import argparse
import copy
import math
import os
import pickle
import subprocess
from functools import partial
from time import perf_counter

import torch
from gradientflow import gradientflow_backprop, gradientflow_kernel, gradientflow_backprop_sgd

from arch import init_arch
from dataset import get_binary_dataset
from kernels import compute_kernels, eigenvectors, kernel_intdim


def loglinspace(step, tau, end=None):
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step * (1 - math.exp(-t / tau)))


def loss_func(args, f, y):
    if args['loss'] == 'hinge':
        return (args['loss_margin'] - args['alpha'] * f * y).relu() / args['alpha']
    if args['loss'] == 'softhinge':
        sp = partial(torch.nn.functional.softplus, beta=args['loss_beta'])
        return sp(args['loss_margin'] - args['alpha'] * f * y) / args['alpha']
    if args['loss'] == 'qhinge':
        return 0.5 * (args['loss_margin'] - args['alpha'] * f * y).relu().pow(2) / args['alpha']


def loss_func_prime(args, f, y):
    if args['loss'] == 'hinge':
        return -((args['loss_margin'] - args['alpha'] * f * y) > 0).double() * y
    if args['loss'] == 'softhinge':
        return -torch.sigmoid(args['loss_beta'] * (args['loss_margin'] - args['alpha'] * f * y)) * y
    if args['loss'] == 'qhinge':
        return -(args['loss_margin'] - args['alpha'] * f * y).relu() * y


def run_kernel(prefix, args, ktrtr, ktetr, ktete, xtr, ytr, xte, yte):
    assert args['f0'] == 1

    assert ktrtr.shape == (len(xtr), len(xtr))
    assert ktetr.shape == (len(xte), len(xtr))
    assert ktete.shape == (len(xte), len(xte))
    assert len(yte) == len(xte)
    assert len(ytr) == len(xtr)

    tau = args['tau_over_h_kernel'] * args['h']
    if args['tau_alpha_crit'] is not None:
        tau *= min(1, args['tau_alpha_crit'] / args['alpha'])

    margin = 0

    checkpoint_generator = loglinspace(args['ckpt_step'], args['ckpt_tau'])
    checkpoint = next(checkpoint_generator)

    wall = perf_counter()
    dynamics = []
    for state, internals in gradientflow_kernel(ktrtr, ytr, tau, partial(loss_func_prime, args), args['max_dgrad'], args['max_dout'] / args['alpha']):
        save_outputs = args['save_outputs']
        save = stop = False

        otr = internals['output']
        grad = internals['gradient']

        if state['step'] == checkpoint:
            checkpoint = next(checkpoint_generator)
            save = True
        if torch.isnan(otr).any():
            save = stop = True
        if wall + args['max_wall_kernel'] < perf_counter():
            save = save_outputs = stop = True
        mind = (args['alpha'] * otr * ytr).min().item()
        if mind > margin:
            margin += 0.5
            save = save_outputs = True
        if mind > args['stop_margin']:
            save = save_outputs = stop = True
        if args['train_kernel'] == 0:
            save = save_outputs = stop = True

        if not save:
            continue

        state['grad_norm'] = grad.norm().item()
        state['wall'] = perf_counter() - wall

        state['train'] = {
            'loss': loss_func(args, otr, ytr).mean().item(),
            'aloss': args['alpha'] * loss_func(args, otr, ytr).mean().item(),
            'err': (otr * ytr <= 0).double().mean().item(),
            'nd': (args['alpha'] * otr * ytr < args['stop_margin']).long().sum().item(),
            'mind': (args['alpha'] * otr * ytr).min().item(),
            'maxd': (args['alpha'] * otr * ytr).max().item(),
            'dfnorm': otr.pow(2).mean().sqrt().item(),
            'alpha_norm': internals['parameters'].norm().item(),
            'outputs': otr.detach().cpu().clone() if save_outputs else None,
            'labels': ytr.cpu() if save_outputs else None,
        }

        # if len(xte) > len(xtr):
        #     from hessian import gradient
        #     a = gradient(f(xtr) @ alpha, f.parameters())
        #     ote = torch.stack([gradient(f(x[None]), f.parameters()) @ a for x in xte])
        # else:
        ote = ktetr @ internals['parameters']

        state['test'] = {
            'loss': loss_func(args, ote, yte).mean().item(),
            'aloss': args['alpha'] * loss_func(args, ote, yte).mean().item(),
            'err': (ote * yte <= 0).double().mean().item(),
            'nd': (args['alpha'] * ote * yte < args['stop_margin']).long().sum().item(),
            'mind': (args['alpha'] * ote * yte).min().item(),
            'maxd': (args['alpha'] * ote * yte).max().item(),
            'dfnorm': ote.pow(2).mean().sqrt().item(),
            'outputs': ote.detach().cpu().clone() if save_outputs else None,
            'labels': yte.cpu() if save_outputs else None,
        }

        print(("[{prefix}] [i={d[step]:d} t={d[t]:.2e} wall={d[wall]:.0f}] [dt={d[dt]:.1e} dgrad={d[dgrad]:.1e} dout={d[dout]:.1e}]" + \
               " [train aL={d[train][aloss]:.2e} err={d[train][err]:.2f} nd={d[train][nd]}/{ptr} mind={d[train][mind]:.3f}]" + \
               " [test aL={d[test][aloss]:.2e} err={d[test][err]:.2f}]").format(prefix=prefix, d=state, ptr=len(xtr), pte=len(xte)), flush=True)
        dynamics.append(state)

        out = {
            'dynamics': dynamics,
            'kernel': None,
        }

        if stop:
            out['kernel'] = {
                'train': {
                    'value': ktrtr.detach().cpu().clone() if args['store_kernel'] == 1 else None,
                    'diag': ktrtr.diag().detach().cpu().clone(),
                    'mean': ktrtr.mean().item(),
                    'std': ktrtr.std().item(),
                    'norm': ktrtr.norm().item(),
                    'intdim': kernel_intdim(ktrtr),
                    'eigenvectors': eigenvectors(ktrtr, ytr),
                },
                'test': {
                    'value': ktete.detach().cpu().clone() if args['store_kernel'] == 1 else None,
                    'diag': ktete.diag().detach().cpu().clone(),
                    'mean': ktete.mean().item(),
                    'std': ktete.std().item(),
                    'norm': ktete.norm().item(),
                    'intdim': kernel_intdim(ktete),
                    'eigenvectors': eigenvectors(ktete, yte),
                },
            }

        yield out
        if stop:
            break


def run_regular(args, f_init, xtr, ytr, xte, yte):

    with torch.no_grad():
        ote0 = f_init(xte)
        otr0 = f_init(xtr)

    if args['f0'] == 0:
        ote0 = torch.zeros_like(ote0)
        otr0 = torch.zeros_like(otr0)

    tau = args['tau_over_h'] * args['h']
    if args['tau_alpha_crit'] is not None:
        tau *= min(1, args['tau_alpha_crit'] / args['alpha'])

    best_test_error = 1
    wall_best_test_error = perf_counter()
    tmp_outputs_index = -1
    margin = 0
    n_changed_dt = 0

    checkpoint_generator = loglinspace(args['ckpt_step'], args['ckpt_tau'])
    checkpoint = next(checkpoint_generator)

    if args['temperature'] == 0.0:
        gradientflow = partial(
            gradientflow_backprop,
            loss=partial(loss_func, args),
            subf0=bool(args['f0']),
            tau=tau,
            chunk=args['chunk'],
            batch=args['bs'],
            max_dgrad=args['max_dgrad'],
            max_dout=args['max_dout'] / args['alpha']
        )
    else:
        gradientflow = partial(
            gradientflow_backprop_sgd,
            loss_function=partial(loss_func, args),
            subf0=bool(args['f0']),
            beta=1.0 / args['temperature'],
            chunk=args['chunk'],
            batch_min=args['batch_min'],
            batch_max=args['batch_max'],
            max_dgrad=args['max_dgrad'],
            max_dout=args['max_dout'] / args['alpha'],
            dt_amplification=args['dt_amp'],
            dt_damping=args['dt_dam'],
        )

    wall = perf_counter()
    dynamics = []
    for state, internals in gradientflow(f_init, xtr, ytr):
        save_outputs = args['save_outputs']
        save = stop = False
        otr = internals['output']
        f = internals['f']
        n_changed_dt += internals['changed_dt']

        if state['step'] == checkpoint:
            checkpoint = next(checkpoint_generator)
            save = True
        if torch.isnan(otr).any():
            save = stop = True
        if wall + args['max_wall'] < perf_counter():
            save = save_outputs = stop = True
        if args['wall_max_early_stopping'] is not None and wall_best_test_error + args['wall_max_early_stopping'] < perf_counter():
            save = save_outputs = stop = True
        if len(otr) == len(xtr):
            mind = (args['alpha'] * otr * ytr).min().item()
            if mind > margin:
                margin += 0.5
                save = save_outputs = True
            if mind > args['stop_margin']:
                save = save_outputs = stop = True
            if (args['ptr'] - (args['alpha'] * otr * ytr < args['stop_margin']).long().sum().item()) / args['ptr'] > args['stop_frac']:
                save = save_outputs = stop = True

        if not save:
            continue

        state['grad_norm'] = internals['gradient'].norm().item()
        state['wall'] = perf_counter() - wall
        state['norm'] = sum(p.norm().pow(2) for p in f.parameters()).sqrt().item()
        state['dnorm'] = sum((p0 - p).norm().pow(2) for p0, p in zip(f_init.parameters(), f.parameters())).sqrt().item()

        if len(otr) == len(xtr) and state['grad_norm'] == 0:
            save = save_outputs = stop = True

        if len(otr) < len(xtr):
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

        test_err = (ote * yte <= 0).double().mean().item()
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
            'loss': loss_func(args, otr, ytr).mean().item(),
            'aloss': args['alpha'] * loss_func(args, otr, ytr).mean().item(),
            'err': (otr * ytr <= 0).double().mean().item(),
            'nd': (args['alpha'] * otr * ytr < args['stop_margin']).long().sum().item(),
            'mind': (args['alpha'] * otr * ytr).min().item(),
            'maxd': (args['alpha'] * otr * ytr).max().item(),
            'mediand': (args['alpha'] * otr * ytr).median().item(),
            'dfnorm': otr.pow(2).mean().sqrt().item(),
            'fnorm': (otr + otr0).pow(2).mean().sqrt().item(),
            'outputs': otr.cpu().clone() if save_outputs else None,
            'labels': ytr.cpu().clone() if save_outputs else None,
        }
        state['test'] = {
            'loss': loss_func(args, ote, yte).mean().item(),
            'aloss': args['alpha'] * loss_func(args, ote, yte).mean().item(),
            'err': test_err,
            'nd': (args['alpha'] * ote * yte < args['stop_margin']).long().sum().item(),
            'mind': (args['alpha'] * ote * yte).min().item(),
            'maxd': (args['alpha'] * ote * yte).max().item(),
            'mediand': (args['alpha'] * ote * yte).median().item(),
            'dfnorm': ote.pow(2).mean().sqrt().item(),
            'fnorm': (ote + ote0).pow(2).mean().sqrt().item(),
            'outputs': ote.cpu().clone() if save_outputs else None,
            'labels': yte.cpu().clone() if save_outputs else None,
        }
        print(
            (
                "[i={d[step]:d} t={d[t]:.2e} wall={d[wall]:.0f}] " + \
                "[{ndt}dt={d[dt]:.1e} dg={d[dgrad]:.1e} do={d[dout]:.1e}] " + \
                "[train aL={d[train][aloss]:.2e} err={d[train][err]:.2f} " + \
                "nd={d[train][nd]}/{p} mind={d[train][mind]:.3f}] " + \
                "[test aL={d[test][aloss]:.2e} err={d[test][err]:.2f}]"
            ).format(d=state, p=len(ytr), ndt=("+{} ".format(n_changed_dt) if n_changed_dt else "")),
            flush=True
        )
        n_changed_dt = 0

        dynamics.append(state)

        out = {
            'dynamics': dynamics,
        }

        if (args['ptr'] - state["train"]["nd"]) / args['ptr'] > args['stop_frac']:
            stop = True

        yield f, out
        if stop:
            break


def run_exp(args, f0, xtr, ytr, xtk, ytk, xte, yte):
    run = {
        'args': args,
        'N': sum(p.numel() for p in f0.parameters()),
        'finished': False,
    }
    wall = None

    if args['init_features_ptr'] == 1:
        parameters = [p for n, p in f0.named_parameters() if 'W{}'.format(args['L']) in n or 'classifier' in n]
        assert parameters
        kernels = compute_kernels(f0, xtr, xte[:len(xtk)], parameters)
        for out in run_kernel('init_features_ptr', args, *kernels, xtr, ytr, xte[:len(xtk)], yte[:len(xtk)]):
            run['init_features_ptr'] = out

            if wall is None or perf_counter() - wall > 120:
                wall = perf_counter()
                yield run
        del kernels

    if args['init_kernel'] == 1:
        init_kernel = compute_kernels(f0, xtk, xte[:len(xtk)])
        for out in run_kernel('init_kernel', args, *init_kernel, xtk, ytk, xte[:len(xtk)], yte[:len(xtk)]):
            run['init_kernel'] = out

    if args['init_kernel_ptr'] == 1:
        init_kernel_ptr = compute_kernels(f0, xtr, xte[:len(xtk)])
        for out in run_kernel('init_kernel_ptr', args, *init_kernel_ptr, xtr, ytr, xte[:len(xtk)], yte[:len(xtk)]):
            run['init_kernel_ptr'] = out
        del init_kernel_ptr

    if args['delta_kernel'] == 1 and args['init_kernel'] == 1:
        init_kernel = init_kernel[0].cpu()
    elif args['delta_kernel'] == 1:
        init_kernel, _, _ = compute_kernels(f0, xtk, xte[:1])
        init_kernel = init_kernel.cpu()
    elif args['init_kernel'] == 1:
        del init_kernel

    if args['regular'] == 1:
        if args['running_kernel']:
            it = iter(args['running_kernel'])
            al = next(it)
        else:
            al = -1
        for f, out in run_regular(args, f0, xtr, ytr, xte, yte):
            run['regular'] = out
            if out['dynamics'][-1]['train']['aloss'] < al * out['dynamics'][0]['train']['aloss']:
                if args['init_kernel_ptr'] == 1:
                    assert len(xtk) >= len(xtr)
                    running_kernel = compute_kernels(f, xtk[:len(xtr)], xte[:len(xtk)])
                    for kout in run_kernel('kernel_ptr {}'.format(al), args, *running_kernel, xtk[:len(xtr)], ytk[:len(xtr)], xte[:len(xtk)], yte[:len(xtk)]):
                        out['dynamics'][-1]['kernel_ptr'] = kout
                    del running_kernel
                if args['init_features_ptr'] == 1:
                    parameters = [p for n, p in f.named_parameters() if 'W{}'.format(args['L']) in n or 'classifier' in n]
                    assert parameters
                    assert len(xtk) >= len(xtr)
                    running_kernel = compute_kernels(f, xtk[:len(xtr)], xte[:len(xtk)], parameters)
                    for kout in run_kernel('features_ptr {}'.format(al), args, *running_kernel, xtk[:len(xtr)], ytk[:len(xtr)], xte[:len(xtk)], yte[:len(xtk)]):
                        out['dynamics'][-1]['features_ptr'] = kout
                    del running_kernel

                out['dynamics'][-1]['state'] = copy.deepcopy(f.state_dict())

                try:
                    al = next(it)
                except StopIteration:
                    al = 0

            if wall is None or perf_counter() - wall > 120:
                wall = perf_counter()
                yield run
        yield run

        if args['final_kernel'] == 1:
            final_kernel = compute_kernels(f, xtk, xte[:len(xtk)])
            if args['final_kernel_ptr'] == 1:
                ktktk, ktetk, ktete = final_kernel
                ktktk = ktktk[:len(xtr)][:, :len(xtr)]
                ktetk = ktetk[:, :len(xtr)]
                final_kernel_ptr = (ktktk, ktetk, ktete)

        elif args['final_kernel_ptr'] == 1:
            final_kernel_ptr = compute_kernels(f, xtk[:len(xtr)], xte[:len(xtk)])

        if args['final_kernel'] == 1:
            for out in run_kernel('final_kernel', args, *final_kernel, xtk, ytk, xte[:len(xtk)], yte[:len(xtk)]):
                run['final_kernel'] = out

                if perf_counter() - wall > 120:
                    wall = perf_counter()
                    yield run
            if args['delta_kernel'] == 0:
                del final_kernel

        if args['final_kernel_ptr'] == 1:
            assert len(xtk) >= len(xtr)
            for out in run_kernel('final_kernel_ptr', args, *final_kernel_ptr, xtk[:len(xtr)], ytk[:len(xtr)], xte[:len(xtk)], yte[:len(xtk)]):
                run['final_kernel_ptr'] = out

                if perf_counter() - wall > 120:
                    wall = perf_counter()
                    yield run
            del final_kernel_ptr

        if args['delta_kernel'] == 1:
            if args['final_kernel'] == 1:
                final_kernel = final_kernel[0].cpu()
            else:
                final_kernel, _, _ = compute_kernels(f, xtk, xte[:1])
                final_kernel = final_kernel.cpu()

            run['delta_kernel'] = {
                'traink': (init_kernel - final_kernel).norm().item(),
            }
            run['delta_kernel']['init'] = {
                'traink': {
                    'value': init_kernel.detach().cpu() if args['store_kernel'] == 1 else None,
                    'diag': init_kernel.diag().detach().cpu().clone(),
                    'mean': init_kernel.mean().item(),
                    'std': init_kernel.std().item(),
                    'norm': init_kernel.norm().item(),
                },
            }
            run['delta_kernel']['final'] = {
                'traink': {
                    'value': final_kernel.detach().cpu() if args['store_kernel'] == 1 else None,
                    'diag': final_kernel.diag().detach().cpu().clone(),
                    'mean': final_kernel.mean().item(),
                    'std': final_kernel.std().item(),
                    'norm': final_kernel.norm().item(),
                },
            }

            del init_kernel, final_kernel

        if args['stretch_kernel'] == 1:
            assert args['save_weights']
            lam = [x["w"][0] / torch.tensor(x["w"][1:]).float().mean() for x in run['regular']["dynamics"]]
            frac = [(args['ptr'] - x["train"]["nd"]) / args['ptr'] for x in run['regular']["dynamics"]]
            for _lam, _frac in zip(lam, frac):
                if _frac > 0.1:
                    lam_star = _lam
                    break
            _xtr = xtr.clone()
            _xte = xte.clone()
            _xtr[:, 1:] = xtr[:, 1:] / lam_star
            _xte[:, 1:] = xte[:, 1:] / lam_star
            stretch_kernel = compute_kernels(f0, _xtr, _xte)
            for out in run_kernel('stretch_kernel', args, *stretch_kernel, _xtr, ytr, _xte, yte):
                run['stretch_kernel'] = out

                if perf_counter() - wall > 120:
                    wall = perf_counter()
                    yield run
            del stretch_kernel

        if args['final_features'] == 1:
            parameters = [p for n, p in f.named_parameters() if 'W{}'.format(args['L']) in n or 'classifier' in n]
            assert parameters
            kernels = compute_kernels(f, xtk, xte[:len(xtk)], parameters)
            for out in run_kernel('final_features', args, *kernels, xtk, ytk, xte[:len(xtk)], yte[:len(xtk)]):
                run['final_features'] = out

                if perf_counter() - wall > 120:
                    wall = perf_counter()
                    yield run
            del kernels

        if args['final_features_ptr'] == 1:
            parameters = [p for n, p in f.named_parameters() if 'W{}'.format(args['L']) in n or 'classifier' in n]
            assert parameters
            assert len(xtk) >= len(xtr)
            kernels = compute_kernels(f, xtk[:len(xtr)], xte[:len(xtk)], parameters)
            for out in run_kernel('final_features_ptr', args, *kernels, xtk[:len(xtr)], ytk[:len(xtr)], xte[:len(xtk)], yte[:len(xtk)]):
                run['final_features_ptr'] = out

                if perf_counter() - wall > 120:
                    wall = perf_counter()
                    yield run
            del kernels

        if args['final_headless'] == 1:
            parameters = [p for n, p in f.named_parameters() if not 'f.W0.' in n and not 'f.conv_stem.w' in n]
            assert len(xtk) >= len(xtr)
            kernels = compute_kernels(f, xtk, xte[:len(xtk)], parameters)
            for out in run_kernel('final_headless', args, *kernels, xtk, ytk, xte[:len(xtk)], yte[:len(xtk)]):
                run['final_headless'] = out

                if perf_counter() - wall > 120:
                    wall = perf_counter()
                    yield run
            del kernels

        if args['final_headless_ptr'] == 1:
            parameters = [p for n, p in f.named_parameters() if not 'f.W0.' in n and not 'f.conv_stem.w' in n]
            assert len(xtk) >= len(xtr)
            kernels = compute_kernels(f, xtk[:len(xtr)], xte[:len(xtk)], parameters)
            for out in run_kernel('final_headless_ptr', args, *kernels, xtk[:len(xtr)], ytk[:len(xtr)], xte[:len(xtk)], yte[:len(xtk)]):
                run['final_headless_ptr'] = out

                if perf_counter() - wall > 120:
                    wall = perf_counter()
                    yield run
            del kernels

    run['finished'] = True
    yield run


def init(args):
    torch.backends.cudnn.benchmark = True
    if args['dtype'] == 'float64':
        torch.set_default_dtype(torch.float64)
    if args['dtype'] == 'float32':
        torch.set_default_dtype(torch.float32)

    [(xte, yte, ite), (xtk, ytk, itk), (xtr, ytr, itr)] = get_binary_dataset(
        args['dataset'],
        (args['pte'], args['ptk'], args['ptr']),
        (args['seed_testset'] + args['pte'], args['seed_kernelset'] + args['ptk'], args['seed_trainset'] + args['ptr']),
        args['d'],
        (args['data_param1'], args['data_param2']),
        args['device'],
        torch.get_default_dtype()
    )

    f, (xtr, xtk, xte) = init_arch((xtr, xtk, xte), **args)

    return f, xtr, ytr, itr, xtk, ytk, itk, xte, yte, ite


def execute(args):
    f, xtr, ytr, itr, xtk, ytk, itk, xte, yte, ite = init(args)

    torch.manual_seed(0)
    for run in run_exp(args, f, xtr, ytr, xtk, ytk, xte, yte):
        run['dataset'] = {
            'test': ite.cpu().clone(),
            'kernel': itk.cpu().clone(),
            'train': itr.cpu().clone(),
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
    parser.add_argument("--seed_testset", type=int, default=0, help="determines the testset, will affect the kernelset and trainset as well")
    parser.add_argument("--seed_kernelset", type=int, default=0, help="determines the kernelset, will affect the trainset as well")
    parser.add_argument("--seed_trainset", type=int, default=0, help="determines the trainset")

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ptr", type=int, required=True)
    parser.add_argument("--ptk", type=int, default=0)
    parser.add_argument("--pte", type=int)
    parser.add_argument("--d", type=int)
    parser.add_argument("--data_param1", type=int,
                        help="Sphere dimension if dataset = Cylinder."
                        "Total number of cells, if dataset = sphere_grid. "
                        "n0 if dataset = signal_1d.")
    parser.add_argument("--data_param2", type=float,
                        help="Stretching factor for non-spherical dimensions if dataset = cylinder."
                        "Number of bins in theta, if dataset = sphere_grid.")

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

    parser.add_argument("--init_kernel", type=int, default=0)
    parser.add_argument("--init_kernel_ptr", type=int, default=0)
    parser.add_argument("--regular", type=int, default=1)
    parser.add_argument('--running_kernel', nargs='+', type=float)
    parser.add_argument("--final_kernel", type=int, default=0)
    parser.add_argument("--final_kernel_ptr", type=int, default=0)
    parser.add_argument("--final_headless", type=int, default=0)
    parser.add_argument("--final_headless_ptr", type=int, default=0)
    parser.add_argument("--init_features_ptr", type=int, default=0)
    parser.add_argument("--final_features", type=int, default=0)
    parser.add_argument("--final_features_ptr", type=int, default=0)
    parser.add_argument("--train_kernel", type=int, default=1)
    parser.add_argument("--store_kernel", type=int, default=0)
    parser.add_argument("--delta_kernel", type=int, default=0)
    parser.add_argument("--stretch_kernel", type=int, default=0)

    parser.add_argument("--save_outputs", type=int, default=0)
    parser.add_argument("--save_state", type=int, default=0)
    parser.add_argument("--save_weights", type=int, default=0)

    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--f0", type=int, default=1)

    parser.add_argument("--tau_over_h", type=float, default=0.0)
    parser.add_argument("--tau_over_h_kernel", type=float)
    parser.add_argument("--tau_alpha_crit", type=float)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch_min", type=int, default=1)
    parser.add_argument("--batch_max", type=int, default=None)
    parser.add_argument("--dt_amp", type=float, default=1.1)
    parser.add_argument("--dt_dam", type=float, default=1.1**3)

    parser.add_argument("--max_wall", type=float, required=True)
    parser.add_argument("--max_wall_kernel", type=float)
    parser.add_argument("--wall_max_early_stopping", type=float)
    parser.add_argument("--chunk", type=int)
    parser.add_argument("--max_dgrad", type=float, default=1e-4)
    parser.add_argument("--max_dout", type=float, default=1e-1)

    parser.add_argument("--loss", type=str, default="softhinge")
    parser.add_argument("--loss_beta", type=float, default=20.0)
    parser.add_argument("--loss_margin", type=float, default=1.0)
    parser.add_argument("--stop_margin", type=float, default=1.0)
    parser.add_argument("--stop_frac", type=float, default=1.0)
    parser.add_argument("--bs", type=int)

    parser.add_argument("--ckpt_step", type=int, default=100)
    parser.add_argument("--ckpt_tau", type=float, default=1e4)

    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    args = args.__dict__

    if args['device'] is None:
        if torch.cuda.is_available():
            args['device'] = 'cuda'
        else:
            args['device'] = 'cpu'

    if args['pte'] is None:
        args['pte'] = args['ptr']

    if args['chunk'] is None:
        args['chunk'] = max(args['ptr'], args['pte'], args['ptk'], 100000)

    if args['max_wall_kernel'] is None:
        args['max_wall_kernel'] = args['max_wall']

    if args['tau_over_h_kernel'] is None:
        args['tau_over_h_kernel'] = args['tau_over_h']

    if args['seed_init'] == -1:
        args['seed_init'] = args['seed_trainset']

    with open(args['output'], 'wb') as handle:
        pickle.dump(args,  handle)

    saved = False
    try:
        for data in execute(args):
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
