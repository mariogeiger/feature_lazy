import argparse
import math
import os
import pickle
import subprocess
import time
from functools import partial, reduce
from itertools import count
import operator

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds
from jax.config import config
from jax.experimental.ode import odeint


def normalize_act(phi):
    with jax.core.eval_context():
        k = jax.random.PRNGKey(0)
        x = jax.random.normal(k, (1_000_000,))
        c = jnp.mean(phi(x)**2)**0.5

    def rho(x):
        return phi(x) / c
    return rho


def mlp(features, phi, x):
    for feat in features:
        d = hk.Linear(
            feat,
            with_bias=False,
            w_init=hk.initializers.RandomNormal()
        )
        x = phi(d(x) / x.shape[-1]**0.5)

    d = hk.Linear(
        1,
        with_bias=False,
        w_init=hk.initializers.RandomNormal()
    )
    x = d(x) / x.shape[-1]
    return x[..., 0]


def mnas(h, act, x):
    def conv2d(c, k, s, x):
        return hk.Conv2D(
            output_channels=c,
            kernel_shape=k,
            stride=s,
            with_bias=False,
            w_init=hk.initializers.RandomNormal(),
        )(x) / (k * x.shape[-1]**0.5)

    def conv2dg(k, s, x):
        return hk.Conv2D(
            output_channels=x.shape[-1],
            kernel_shape=k,
            stride=s,
            feature_group_count=x.shape[-1],
            with_bias=False,
            w_init=hk.initializers.RandomNormal(),
        )(x) / (k)

    x = act(conv2d(round(4 * h), 5, 2, x))
    x = act(conv2dg(5, 1, x))
    x = act(conv2d(round(2 * h), 1, 1, x))

    def inverted_residual(out_chs, k, s, x):
        in_chs = x.shape[-1]
        mid_chs = round(in_chs * 3.0)

        residual = x
        x = act(conv2d(mid_chs, 1, 1, x))
        x = act(conv2dg(k, s, x))
        x = conv2d(out_chs, 1, 1, x)

        if residual.shape == x.shape:
            x = (x + residual) / 2**0.5
        else:
            x = act(x)
        return x

    x = inverted_residual(round(h), 5, 2, x)
    x = inverted_residual(round(h), 5, 1, x)
    x = inverted_residual(round(3 * h), 5, 2, x)
    x = inverted_residual(round(3 * h), 5, 1, x)

    x = act(conv2d(round(20 * h), 1, 1, x))

    x = jnp.mean(x, axis=(1, 2))
    x = hk.Linear(
        output_size=1,
        with_bias=False,
        w_init=hk.initializers.RandomNormal(),
    )(x) / (x.shape[-1])
    return x[..., 0]


def dataset(dataset, seed_trainset, seed_testset, ptr, pte, **args):
    if dataset in ['stripe', 'sign']:
        d = args['d']
        xtr = jax.random.normal(jax.random.PRNGKey(seed_trainset), (ptr, d))
        xte = jax.random.normal(jax.random.PRNGKey(seed_testset), (pte, d))

        if dataset == 'stripe':
            def y(x):
                return 2 * (x[:, 0] > -0.3) * (x[:, 0] < 1.18549) - 1

        elif dataset == 'sign':
            def y(x):
                return 2 * (x[:, 0] > 0) - 1

        return xtr, xte, y(xtr), y(xte)

    if dataset == 'mnist_parity':
        ds = tfds.load("mnist", split='train+test')

        def x(images):
            return jnp.array(images).astype(jnp.float32) / 255 * 3.0

        def y(labels):
            return jnp.array([
                -1.0 if y % 2 == 1 else 1.0
                for y in labels
            ])

    if dataset == 'cifar_animal':
        ds = tfds.load("cifar10", split='train+test')

        def x(images):
            return jnp.array(images).astype(jnp.float32) / 255 * 1.87

        def y(labels):
            return jnp.array([
                -1.0 if y in [0, 1, 8, 9] else 1.0
                for y in labels
            ])

    ds = ds.shuffle(len(ds), seed=seed_trainset, reshuffle_each_iteration=False)
    dtr = ds.take(ptr)

    dte = ds.skip(ptr)
    dte = dte.shuffle(len(dte), seed=seed_testset, reshuffle_each_iteration=False)
    dte = dte.take(pte)

    dtr = next(dtr.batch(len(dtr)).as_numpy_iterator())
    xtr, ytr = x(dtr['image']), y(dtr['label'])

    dte = next(dte.batch(len(dte)).as_numpy_iterator())
    xte, yte = x(dte['image']), y(dte['label'])

    return xtr, xte, ytr, yte


def sgd_drift(f, loss, n, bs, dt, key, w, out0, x, y):
    assert n * bs <= x.shape[0]

    w2 = odeint(
        lambda w, t: jax.grad(lambda w: -jnp.mean(loss(f(w, x) - out0, y)), 0)(w),
        w, jnp.array([0, 2 * dt])
    )
    dw_gf = jax.tree_map(lambda a, b: a[1] - b, w2, w)

    i = jax.random.permutation(key, x.shape[0])
    i = i[:n * bs]
    o01 = out0[i].reshape((n, bs) + out0.shape[1:])
    x1 = x[i].reshape((n, bs) + x.shape[1:])
    y1 = y[i].reshape((n, bs) + y.shape[1:])

    dloss = jax.grad(lambda w, o0, x, y: jnp.mean(loss(f(w, x) - o0, y)), 0)

    g1 = jax.vmap(dloss, (None, 0, 0, 0), 0)(w, o01, x1, y1)
    w1 = jax.tree_map(lambda a, b: a - dt * b, w, g1)

    g2 = jax.vmap(dloss, (0, None, None, None), 0)(w1, out0, x, y)
    dw_sgd = jax.tree_map(lambda a, b: jnp.mean(-dt * (a + b), 0), g1, g2)

    return jax.tree_leaves(jax.tree_map(lambda a, b: jnp.sum((a - b)**2), dw_sgd, dw_gf))


def mean_var_grad(f, loss, w, out0, x, y):
    out, j = jax.vmap(jax.value_and_grad(f, 0), (None, 0), 0)(w, x)
    j = jnp.concatenate([
        jnp.reshape(x, (x.shape[0], reduce(operator.mul, x.shape[1:], 1)))
        for x in jax.tree_leaves(j)
    ], 1)  # [x, w]
    # j[i, j] = d f(w, x_i) / d w_j
    mean_f = jnp.mean(j, 0)
    var_f = jnp.mean(jnp.sum((j - mean_f)**2, 1))

    # kernel[mu,nu] = sum_j j[mu,j] j[nu,j]
    kernel = j @ j.T

    dl = jax.vmap(jax.grad(loss, 0), (0, 0), 0)
    lj = dl(out - out0, y)[:, None] * j
    mean_l = jnp.mean(lj, 0)
    var_l = jnp.mean(jnp.sum((lj - mean_l)**2, 1))

    return jnp.sum(mean_f**2), var_f, jnp.sum(mean_l**2), var_l, kernel


def sgd(f, loss, bs, dt, key, w, out0, xtr, ytr):
    key, k = jax.random.split(key)
    i = jax.random.permutation(k, xtr.shape[0])[:bs]
    x = xtr[i]
    y = ytr[i]
    o0 = out0[i]
    lo, g = jax.value_and_grad(lambda w: jnp.mean(loss(f(w, x) - o0, y)))(w)
    w = jax.tree_multimap(lambda w, g: w - dt * g, w, g)
    return key, w, lo


def sgd_until(f, loss, bs, dt, key, w, out0, xtr, ytr, last_loss, target_loss, num):
    def cond(x):
        _key, _w, last_loss, i = x
        return (target_loss < last_loss) & (i < num) & jnp.isfinite(last_loss)

    def body(x):
        key, w, last_loss, i = x
        key, w, batch_loss = sgd(f, loss, bs, dt, key, w, out0, xtr, ytr)
        current_loss = ((xtr.shape[0] - bs) * last_loss + bs * batch_loss) / xtr.shape[0]
        return key, w, current_loss, i + 1

    return jax.lax.while_loop(cond, body, (key, w, last_loss, 0))


def hinge(o, y):
    return jax.nn.relu(1.0 - o * y)


def qhinge(o, y):
    return jax.nn.relu(1.0 - o * y)**2


def sus(x):
    return jnp.where(x > 0.0, jnp.exp(-1.0 / jnp.where(x > 0.0, x, 1.0)), 0.0)


def srelu(x):
    return sus(2.0 * x) * x


def shinge(o, y):
    return srelu(1.0 - o * y)


def train(
    f, w0, xtr, xte, ytr, yte, bs, dt, seed_batch, alpha, loss,
    ckpt_step, ckpt_grad_stats, ckpt_kernels,
    max_wall, max_step, mind_stop, **args
):
    key_batch = jax.random.PRNGKey(seed_batch)

    if loss == "hinge":
        loss_fun = hinge
    elif loss == "softhinge":
        loss_fun = shinge
    elif loss == "quadhinge":
        loss_fun = qhinge
    else:
        raise ValueError("{loss} is not a valid choice of loss")

    loss = lambda o, y: loss_fun(alpha * o, y) / alpha

    jit_sgd_until = jax.jit(partial(sgd_until, f, loss, bs, dt))
    jit_mean_var_grad = jax.jit(partial(mean_var_grad, f, loss))
    jit_sgd_drift = jax.jit(partial(sgd_drift, f, loss, 32, bs, dt))

    @jax.jit
    def jit_le(w, out0, x, y):
        out = jnp.concatenate([
            f(w, x[i: i + 1024])
            for i in range(0, x.shape[0], 1024)
        ])
        pred = out - out0
        return pred, jnp.mean(loss(pred, y)), jnp.mean((pred * y <= 0 | ~jnp.isfinite(pred)))

    out0tr = jnp.concatenate([f(w0, xtr[i: i + 1024]) for i in range(0, xtr.shape[0], 1024)])
    out0te = jnp.concatenate([f(w0, xte[i: i + 1024]) for i in range(0, xte.shape[0], 1024)])
    pred, l0, _ = jit_le(w0, out0tr, xtr, ytr)

    _, _, _, _, kernel_tr0 = jit_mean_var_grad(w0, out0tr[:ckpt_grad_stats], xtr[:ckpt_grad_stats], ytr[:ckpt_grad_stats])
    _, _, _, _, kernel_te0 = jit_mean_var_grad(w0, out0te[:ckpt_grad_stats], xte[:ckpt_grad_stats], yte[:ckpt_grad_stats])

    ckpt_loss = 2.0**jnp.arange(-24, -9, 0.5)
    ckpt_loss = l0 * jnp.concatenate([ckpt_loss, jnp.arange(2**-9, 1, 2**-9), 1 - ckpt_loss[::-1]])

    ckpt_loss_grad = 2.0**jnp.arange(-20, -2, 0.5)
    ckpt_loss_grad = l0 * jnp.concatenate([ckpt_loss_grad, jnp.arange(2**-2, 1.0, 2**-4), jnp.array([1.0 - 2**-20])])

    target_loss = l0
    current_loss = l0
    target_loss_for_ckpt_grad = l0

    dynamics = []
    w = w0
    wall0 = time.perf_counter()
    wall_print = 0
    wall_yield = 0
    wall_ckpt = 0
    t = 0
    step = 0

    for _ in count():

        total_step = 0
        while True:
            key_batch, w, current_loss, num_step = jit_sgd_until(key_batch, w, out0tr, xtr, ytr, current_loss, target_loss, ckpt_step)
            t += dt * num_step
            step += num_step
            total_step += num_step

            pred, l, err = jit_le(w, out0tr, xtr, ytr)
            current_loss = l

            if ckpt_step <= total_step or (not jnp.isfinite(current_loss)):
                break

            if current_loss <= target_loss:
                if ckpt_loss[0] < current_loss:
                    target_loss = ckpt_loss[ckpt_loss < current_loss][-1]
                else:
                    target_loss = 0.0
                break

        key_batch.block_until_ready()
        wckpt = time.perf_counter()

        stop = False

        if l == 0.0:
            stop = True

        if ckpt_step <= total_step and alpha * jnp.min(pred * ytr) >= mind_stop:
            stop = True

        if not jnp.isfinite(l):
            stop = True

        if time.perf_counter() - wall0 > max_wall:
            stop = True

        if step > max_step:
            stop = True

        save_grad = False
        if ckpt_grad_stats and current_loss <= target_loss_for_ckpt_grad:
            save_grad = True
            if ckpt_loss_grad[0] < current_loss:
                target_loss_for_ckpt_grad = ckpt_loss_grad[ckpt_loss_grad < current_loss][-1]
            else:
                target_loss_for_ckpt_grad = 0.0

        mean_f, var_f, mean_l, var_l, kernel, kernel_change, drift = [None] * 7
        if save_grad:
            mean_f, var_f, mean_l, var_l, kernel = jit_mean_var_grad(w, out0tr[:ckpt_grad_stats], xtr[:ckpt_grad_stats], ytr[:ckpt_grad_stats])
            kernel_change = jnp.mean((kernel - kernel_tr0)**2)
            kernel_norm = jnp.mean(kernel**2)

            drift = jit_sgd_drift(key_batch, w, out0tr, xtr, ytr)

        train = dict(
            loss=l,
            aloss=alpha * l,
            err=err,
            mind=jnp.min(pred * ytr),
            nd=jnp.sum(alpha * pred * ytr < 1.0),
            drift=drift,
            grad_f_norm=mean_f,
            grad_f_var=var_f,
            grad_l_norm=mean_l,
            grad_l_var=var_l,
            pred=(pred if stop else None),
            label=(ytr if stop else None),
            kernel=(kernel if ckpt_kernels else None),
            kernel_change=kernel_change,
            kernel_norm=kernel_norm,
        )
        del l, err

        if save_grad:
            mean_f, var_f, mean_l, var_l, kernel = jit_mean_var_grad(w, out0te[:ckpt_grad_stats], xte[:ckpt_grad_stats], yte[:ckpt_grad_stats])
            kernel_change = jnp.mean((kernel - kernel_te0)**2)
            kernel_norm = jnp.mean(kernel**2)

        pred, l, err = jit_le(w, out0te, xte, yte)

        test = dict(
            loss=l,
            aloss=alpha * l,
            err=err,
            mind=jnp.min(pred * yte),
            nd=jnp.sum(alpha * pred * yte < 1.0),
            grad_f_norm=mean_f,
            grad_f_var=var_f,
            grad_l_norm=mean_l,
            grad_l_var=var_l,
            pred=(pred if stop else None),
            label=(yte if stop else None),
            kernel=(kernel if ckpt_kernels else None),
            kernel_change=kernel_change,
            kernel_norm=kernel_norm,
        )
        del l, err

        state = dict(
            t=t,
            step=step,
            wall=time.perf_counter() - wall0,
            weights_norm=[jnp.sum(x**2) for x in jax.tree_leaves(w)],
            delta_weights_norm=[jnp.sum((x - x0)**2) for x, x0 in zip(jax.tree_leaves(w), jax.tree_leaves(w0))],
            train=train,
            test=test,
        )
        dynamics.append(jax.tree_map(np.asarray, state))

        if time.perf_counter() - wall_print > 0.1 or stop:
            wall_print = time.perf_counter()

            print((
                f"[{step} t={t:.2e} w={state['wall']:.0f} "
                f"s={len(dynamics)} "
                f"ckpt={100 * wall_ckpt / state['wall']:.0f}%] "
                f"[train aL={alpha * state['train']['loss']:.2e} err={state['train']['err']:.2f} mind={alpha * state['train']['mind']:.2f}] "
                f"[test aL={alpha * state['test']['loss']:.2e} err={state['test']['err']:.2f}]"
            ), flush=True)

        if time.perf_counter() - wall_yield > 10.0 or stop:
            wall_yield = time.perf_counter()
            yield dynamics

        del state

        wall_ckpt += time.perf_counter() - wckpt

        if stop:
            return


def init(arch, h, L, act, seed_init, **args):
    if act == 'silu':
        act = jax.nn.silu
    if act == 'gelu':
        act = jax.nn.gelu
    if act == 'relu':
        act = jax.nn.relu

    act = normalize_act(act)

    xtr, xte, ytr, yte = dataset(**args)
    print('dataset generated', flush=True)

    if arch == 'mlp':
        model = hk.without_apply_rng(hk.transform(
            lambda x: mlp([h] * L, act, x)
        ))

        xtr = xtr.reshape(xtr.shape[0], -1)
        xte = xte.reshape(xte.shape[0], -1)

    if arch == 'mnas':
        model = hk.without_apply_rng(hk.transform(
            lambda x: mnas(h, act, x)
        ))

    print(f'xtr.shape={xtr.shape} xte.shape={xte.shape}', flush=True)

    w = model.init(jax.random.PRNGKey(seed_init), xtr)
    print('network initialized', flush=True)

    return model, w, xtr, xte, ytr, yte


def execute(**args):
    print(f"device={jnp.ones(3).device_buffer.device()} dtype={jnp.ones(3).dtype}", flush=True)

    model, w, xtr, xte, ytr, yte = init(**args)

    darch = dict(
        params_shapes=[p.shape for p in jax.tree_leaves(w)],
        params_sizes=[p.size for p in jax.tree_leaves(w)],
    )

    for d in train(model.apply, w, xtr, xte, ytr, yte, **args):
        yield dict(
            arch=darch,
            sgd=dict(dynamics=d),
            finished=False,
        )

    yield dict(
        arch=darch,
        sgd=dict(dynamics=d),
        finished=True,
    )


def main():
    git = {
        'log': subprocess.getoutput('git log --format="%H" -n 1 -z'),
        'status': subprocess.getoutput('git status -z'),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_init", default=0)
    parser.add_argument("--seed_batch", default=0)
    parser.add_argument("--seed_trainset", default=-1)
    parser.add_argument("--seed_testset", default=-2)

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ptr", type=int, required=True)
    parser.add_argument("--pte", type=int, required=True)
    parser.add_argument("--d", type=int)

    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--act", type=str, required=True)
    parser.add_argument("--act_beta", type=float, default=1.0)
    parser.add_argument("--L", type=int)
    parser.add_argument("--h", type=int, required=True)

    parser.add_argument("--loss", type=str, default="hinge")
    parser.add_argument("--alpha", type=float, required=True)

    parser.add_argument("--bs", type=int, required=True)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--temp", type=float)

    parser.add_argument("--max_wall", type=float, required=True)
    parser.add_argument("--max_step", type=float, default=np.inf)
    parser.add_argument("--mind_stop", type=float, default=1.0)

    parser.add_argument("--ckpt_step", type=int, default=4096)
    parser.add_argument("--ckpt_grad_stats", type=int, default=0)
    parser.add_argument("--ckpt_kernels", type=int, default=0)

    parser.add_argument("--dtype", type=str, default="f32")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args().__dict__

    if args['dtype'] == "f64":
        config.update("jax_enable_x64", True)
    else:
        config.update("jax_enable_x64", False)
        assert args['dtype'] == "f32"

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
