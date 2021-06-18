import argparse
import math
import os
import pickle
import subprocess
import time
from functools import partial
from itertools import count
from typing import Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


def normalize_act(phi):
    k = jax.random.PRNGKey(0)
    x = jax.random.normal(k, (1_000_000,))
    c = jnp.mean(phi(x)**2)**0.5

    def rho(x):
        return phi(x) / c
    return rho


class MLP(nn.Module):
    features: Sequence[int]
    phi: Callable

    @nn.compact
    def __call__(self, x):
        for feat in self.features:
            d = nn.Dense(feat, kernel_init=jax.random.normal, use_bias=False)
            x = self.phi(d(x) / x.shape[-1]**0.5)

        d = nn.Dense(1, kernel_init=jax.random.normal, use_bias=False)
        x = d(x) / x.shape[-1]
        return x[..., 0]


def mean_var_grad(f, loss, w, out0, x, y):
    j = jax.jacobian(f.apply, 0)(w, x)
    j = jnp.concatenate([jnp.reshape(x, (x.shape[0], math.prod(x.shape[1:]))) for x in jax.tree_leaves(j)], 1)  # [x, w]
    # j[i, j] = d f(w, x_i) / d w_j
    mean_f = jnp.mean(j, 0)
    var_f = jnp.mean(jnp.sum((j - mean_f)**2, 1))

    dl = jax.vmap(jax.grad(loss, 0), (0, 0), 0)
    j = dl(f.apply(w, x) - out0, y)[:, None] * j
    mean_l = jnp.mean(j, 0)
    var_l = jnp.mean(jnp.sum((j - mean_l)**2, 1))

    return jnp.sum(mean_f**2), var_f, jnp.sum(mean_l**2), var_l


def dataset(dataset, seed_trainset, seed_testset, ptr, pte, d, **args):
    xtr = jax.random.normal(jax.random.PRNGKey(seed_trainset), (ptr, d))
    xte = jax.random.normal(jax.random.PRNGKey(seed_testset), (pte, d))

    if dataset == 'stripe':
        def y(x):
            return 2 * (x[:, 0] > -0.3) * (x[:, 0] < 1.18549) - 1

    return xtr, xte, y(xtr), y(xte)


def sgd(f, loss, bs, key, w, out0, xtr, ytr):
    i = jax.random.permutation(key, xtr.shape[0])[:bs]
    x = xtr[i]
    y = ytr[i]
    o0 = out0[i]
    return jax.grad(lambda w: jnp.mean(loss(f.apply(w, x) - o0, y)))(w)


def hinge(alpha, o, y):
    return nn.relu(1.0 - alpha * o * y) / alpha


def train(f, w0, xtr, xte, ytr, yte, bs, dt, seed_batch, alpha, ckpt_factor, ckpt_loss, ckpt_grad_stats, max_wall, **args):
    key_batch = jax.random.PRNGKey(seed_batch)

    loss = partial(hinge, alpha)

    jit_sgd = jax.jit(partial(sgd, f, loss, bs))
    jit_mean_var_grad = jax.jit(partial(mean_var_grad, f, loss))

    @jax.jit
    def jit_le(w, out0, x, y):
        pred = f.apply(w, x) - out0
        return jnp.mean(loss(pred, y)), jnp.mean(pred * y <= 0)

    out0tr = f.apply(w0, xtr)
    out0te = f.apply(w0, xte)
    l0, _ = jit_le(w0, out0tr, xtr, ytr)
    _, _ = jit_le(w0, out0te, xte, yte)

    dynamics = []
    w = w0
    wall0 = time.perf_counter()
    wall_print = 0
    save_step = 0
    t = 0
    for step in count():

        key_batch, k = jax.random.split(key_batch)
        g = jit_sgd(k, w, out0tr, xtr, ytr)

        if step >= save_step:
            save_step += ckpt_factor * step

            l, err = jit_le(w, out0tr, xtr, ytr)

            if l < (1 - ckpt_loss) * l0 or step == 0:
                mean_f, var_f, mean_l, var_l = jit_mean_var_grad(w, out0tr[:ckpt_grad_stats], xtr[:ckpt_grad_stats], ytr[:ckpt_grad_stats])

                train = dict(
                    loss=float(l),
                    err=float(err),
                    grad_f_norm=float(mean_f),
                    grad_f_var=float(var_f),
                    grad_l_norm=float(mean_l),
                    grad_l_var=float(var_l),
                )
                del l, err

                mean_f, var_f, mean_l, var_l = jit_mean_var_grad(w, out0te[:ckpt_grad_stats], xte[:ckpt_grad_stats], yte[:ckpt_grad_stats])
                l, err = jit_le(w, out0tr, xte, yte)

                test = dict(
                    loss=float(l),
                    err=float(err),
                    grad_f_norm=float(mean_f),
                    grad_f_var=float(var_f),
                    grad_l_norm=float(mean_l),
                    grad_l_var=float(var_l),
                )
                del l, err

                state = dict(
                    t=t,
                    step=step,
                    wall=time.perf_counter() - wall0,
                    weights_norm=[float(jnp.sum(x**2)) for x in jax.tree_leaves(w)],
                    delta_weights_norm=[float(jnp.sum((x - x0)**2)) for x, x0 in zip(jax.tree_leaves(w), jax.tree_leaves(w0))],
                    train=train,
                    test=test,
                )
                dynamics.append(state)

                if time.perf_counter() - wall_print > 1.0:
                    wall_print = time.perf_counter()

                    print((
                        f"[{step} t={t:.2e}] "
                        f"[train aL={alpha * state['train']['loss']:.2e} err={state['train']['err']:.2f}] "
                        f"[test aL={alpha * state['test']['loss']:.2e} err={state['test']['err']:.2f}]"
                    ), flush=True)

                    yield dynamics

                if state['train']['loss'] == 0.0:
                    break

                if state['wall'] > max_wall:
                    break

                del state

        w = jax.tree_multimap(lambda w, g: w - dt * g, w, g)
        t += dt

    yield dynamics


def execute(arch, h, L, act, seed_init, **args):
    if act == 'silu':
        act = nn.silu
    if act == 'relu':
        act = nn.relu

    act = normalize_act(act)

    if arch == 'mlp':
        f = MLP([h] * L, act)

    xtr, xte, ytr, yte = dataset(**args)

    w = f.init(jax.random.PRNGKey(seed_init), xtr[:2])

    for d in train(f, w, xtr, xte, ytr, yte, **args):
        yield dict(
            dynamics=d,
        )


def main():
    git = {
        'log': subprocess.getoutput('git log --format="%H" -n 1 -z'),
        'status': subprocess.getoutput('git status -z'),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_init", default=0)
    parser.add_argument("--seed_batch", default=0)
    parser.add_argument("--seed_testset", default=0)
    parser.add_argument("--seed_trainset", default=0)

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ptr", type=int, required=True)
    parser.add_argument("--pte", type=int, required=True)
    parser.add_argument("--d", type=int, required=True)

    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--act", type=str, required=True)
    parser.add_argument("--act_beta", type=float, default=1.0)
    parser.add_argument("--L", type=int)
    parser.add_argument("--h", type=int, required=True)

    parser.add_argument("--alpha", type=float, required=True)

    parser.add_argument("--bs", type=int, required=True)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--temp", type=float)

    parser.add_argument("--max_wall", type=float, required=True)

    parser.add_argument("--ckpt_factor", type=float, default=1e-3)
    parser.add_argument("--ckpt_loss", type=float, default=1e-3)
    parser.add_argument("--ckpt_grad_stats", type=int, default=0)

    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args().__dict__

    # dt and derivatives
    assert (args['dt'] is not None) + (args['temp'] is not None) == 1

    if args['temp'] is not None:
        args['dt'] = args['temp'] * args['alpha'] * args['h'] * args['bs']

    if args['temp'] is None:
        args['temp'] = args['dt'] / (args['alpha'] * args['h'] * args['bs'])
    # end

    if args['seed_init'] == 'seed_trainset':
        args['seed_init'] = args['seed_trainset']

    if args['seed_batch'] == 'seed_init':
        args['seed_batch'] = args['seed_init']

    args['seed_init'] = int(args['seed_init'])
    args['seed_batch'] = int(args['seed_batch'])
    args['seed_trainset'] = int(args['seed_trainset'])
    args['seed_testset'] = int(args['seed_testset'])

    with open(args['output'], 'wb') as handle:
        pickle.dump(args,  handle)

    saved = False
    try:
        for data in execute(**args):
            data['git'] = git
            data['args'] = args
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
