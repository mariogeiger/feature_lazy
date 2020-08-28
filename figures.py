# pylint: disable=no-member, missing-docstring, invalid-name, line-too-long
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import torch

from grid import load


def mean(x):
    x = list(x)
    return sum(x) / len(x)


def median(x):
    x = sorted(list(x))
    return x[len(x) // 2]


def triangle(a, b, c, d=None, slope=None, other=False, color=None, fmt="{:.2f}", textpos=None):
    import math

    if slope is not None and d is None:
        d = math.exp(math.log(c) + slope * (math.log(b) - math.log(a)))
    if slope is not None and c is None:
        c = math.exp(math.log(d) - slope * (math.log(b) - math.log(a)))
    if color is None:
        color = "k"

    plt.plot([a, b], [c, d], color=color)
    if other:
        plt.plot([a, b], [c, c], color=color)
        plt.plot([b, b], [c, d], color=color)
    else:
        plt.plot([a, b], [d, d], color=color)
        plt.plot([a, a], [c, d], color=color)

    s = (math.log(d) - math.log(c)) / (math.log(b) - math.log(a))
    if other:
        x = math.exp(0.7 * math.log(b) + 0.3 * math.log(a))
        y = math.exp(0.7 * math.log(c) + 0.3 * math.log(d))
    else:
        x = math.exp(0.7 * math.log(a) + 0.3 * math.log(b))
        y = math.exp(0.7 * math.log(d) + 0.3 * math.log(c))
    if textpos:
        x = textpos[0]
        y = textpos[1]
    plt.annotate(
        fmt.format(s), (x, y), horizontalalignment="center", verticalalignment="center"
    )
    return s


def nd(x, a):
    assert not torch.isnan(x["outputs"]).any()
    return (a * x["outputs"] * x["labels"] < 1).nonzero().numel()


def err(x):
    assert not torch.isnan(x["outputs"]).any()
    f = x["outputs"]
    y = x["labels"]
    if f.dim() == 2 and y.dtype == torch.long:
        e = f.argmax(1) != y
    if f.dim() == 1 and y.dtype == f.dtype:
        e = f * y <= 0
    return e.double().mean().item()


def enserr(xs):
    f = mean(x["outputs"] for x in xs)
    y = xs[0]["labels"]
    assert all((x["labels"] == y).all() for x in xs)
    if f.dim() == 2 and y.dtype == torch.long:
        e = f.argmax(1) != y
    if f.dim() == 1 and y.dtype == f.dtype:
        e = f * y <= 0
    return e.double().mean().item()


def var(outs, alpha):
    otr = alpha * torch.stack(outs)
    return otr.sub(otr.mean(0)).pow(2).mean(1).sum(0).item() / (otr.size(0) - 1)


def texnum(x, mfmt="{}"):
    m, e = "{:e}".format(x).split("e")
    m, e = float(m), int(e)
    mx = mfmt.format(m)
    if e == 0:
        if m == 1:
            return "1"
        return mx
    ex = r"10^{{{}}}".format(e)
    if m == 1:
        return ex
    return r"{}\;{}".format(mx, ex)


def logfilter(x, y, num):
    import numpy as np
    import scipy.ndimage

    x = np.array(x)
    y = np.array(y)
    x = np.log(x)
    xi = np.linspace(min(x), max(x), num)
    yi = np.interp(xi, x, y)
    yf = scipy.ndimage.filters.gaussian_filter1d(yi, 2)
    return np.exp(xi), yf


def yavg(xi, x, y):
    import numpy as np

    xi = np.array(xi)
    xmin = min(np.min(x) for x in x)
    xmax = min(np.max(x) for x in x)
    xi = xi[np.logical_and(xmin < xi, xi < xmax)]
    y = [np.interp(xi, np.array(x), np.array(y)) for x, y in zip(x, y)]
    y = np.mean(y, axis=0)
    return xi, y



# plt.gca().yaxis.set_major_formatter(format_percent)
@ticker.FuncFormatter
def format_percent(x):
    x = 100 * x
    if x % 1 > 0.05:
        return r"${:.1f}\%$".format(x)
    else:
        return r"${:.0f}\%$".format(x)


# git checkout c76419e6abbc4b5535a62e4fe76e2d5e31bad948
# python -m grid MMNASswish_p_6 --n 16 "srun --partition gpu --qos gpu --gres gpu:1 --time 16:00:00 --mem 12G python main.py --arch mnas --act swish --dataset mnist --loss softhinge --loss_beta 3 --max_dgrad 1e-3 --max_dout 0.1 --tau_over_h 1 --tau_over_h_kernel 0 --max_wall 39600 --max_wall_kernel 7200 --h 10 --store_kernel 0 --train_kernel 1 --init_kernel_ptr 1 --final_kernel_ptr 1 --init_features_ptr 1 --final_features_ptr 1 --pte 10000 --ptk 7000" --seed_init 0 1 2 3 4 --ptr 42 51 61 73 87 105 126 151 181 217 261 313 376 451 541 649 779 935 1122 1346 1615 1938 2326 2791 3349 4019 4823 5787 6944 --alpha 2e-3

runs = load("MMNASswish_p_6")
print(len(runs))
runs = [r for r in runs if r["finished"]]
print(len(runs))

ps = sorted({r["args"].ptr for r in runs})

print([len([1 for r in runs if r["args"].ptr == p]) for p in ps])

plt.figure(figsize=(5, 4), dpi=120)

[line1] = plt.plot(
    ps,
    [
        mean(
            r["regular"]["dynamics"][-1]["test"]["err"]
            for r in runs
            if r["args"].ptr == p
        )
        for p in ps
    ],
    label="neural network",
)

[line2] = plt.plot(
    ps,
    [
        mean(
            r["final_kernel_ptr"]["dynamics"][-1]["test"]["err"]
            for r in runs
            if r["args"].ptr == p
        )
        for p in ps
    ],
    label=r"full kernel at",
)

plt.plot(
    ps,
    [
        mean(
            r["init_kernel_ptr"]["dynamics"][-1]["test"]["err"]
            for r in runs
            if r["args"].ptr == p
        )
        for p in ps
    ],
    "--",
    color=line2.get_color(),
    label=r"full kernel at",
)

[line3] = plt.plot(
    ps,
    [
        mean(
            r["final_features_ptr"]["dynamics"][-1]["test"]["err"]
            for r in runs
            if r["args"].ptr == p
        )
        for p in ps
    ],
    label=r"last layer kernel at",
)

plt.plot(
    ps,
    [
        mean(
            r["init_features_ptr"]["dynamics"][-1]["test"]["err"]
            for r in runs
            if r["args"].ptr == p
        )
        for p in ps
    ],
    "--",
    color=line3.get_color(),
    label=r"last layer kernel at",
)


plt.legend(
    [
        line1,
        line2,
        line3,
        Line2D([0], [0], color="k"),
        Line2D([0], [0], color="k", linestyle="--"),
    ],
    [
        "network",
        "full kernel",
        "last layer kernel",
        r"after training",
        r"at initialization",
    ],
)

plt.xscale("log")
plt.yscale("log")

plt.xlabel(r"$p$")
plt.ylabel("test error")

plt.xlim(ps[0], ps[-1])
triangle(5e2, 2.5e3, 7e-2, slope=-1 / 2, fmt="-1/2")
triangle(2e3, 6e3, 1e-1, slope=-1 / 3, fmt='-1/3')

plt.tight_layout()
plt.savefig("MMNASswish_p.pgf")
