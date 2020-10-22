# pylint: disable=missing-docstring, invalid-name, line-too-long
import argparse
import math

from grid.exec import exec_blocking
from grid import load


def round_mantissa(x, n):
    m = round(math.log2(x))
    return round(2**(n-m) * x) * 2**(m-n)


def mean(z):
    return sum(z) / len(z)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("log_dir", type=str)
    parser.add_argument("cmd", type=str)

    parser.add_argument("--alpha_min", type=float)
    parser.add_argument("--alpha_max", type=float)
    parser.add_argument("--tol", type=int)
    parser.add_argument("--c", type=float)
    parser.add_argument("--h", type=int)

    parser.add_argument("--seed_init", type=int, nargs='+')

    args = parser.parse_args()

    def get_value(r):
        return math.log10(r['delta_kernel']['traink'] / r['delta_kernel']['init']['traink']['norm'])

    obj_value = math.log10(args.c)

    alpha_min = args.alpha_min
    alpha_max = args.alpha_max

    runs = load(args.log_dir, pred_args=lambda a: a.h == args.h and a.seed_init in args.seed_init, pred_run=lambda r: r['finished'])
    als = sorted({r['args'].alpha for r in runs})
    als = [a for a in als if len([r for r in runs if r['args'].alpha == a]) == len(args.seed_init)]
    alpha_max = min([args.alpha_max] + [a for a in als if mean([get_value(r) for r in runs if r['args'].alpha == a]) < obj_value])
    alpha_min = max([args.alpha_min] + [a for a in als if mean([get_value(r) for r in runs if r['args'].alpha == a]) > obj_value])

    while True:
        alpha = math.sqrt(alpha_min * alpha_max)
        alpha = round_mantissa(alpha, args.tol)

        if alpha in [alpha_min, alpha_max]:
            break

        print("{} < {} < {}".format(alpha_min, alpha, alpha_max))

        rs = [
            exec_blocking(args.log_dir, args.cmd, (('h', args.h), ('alpha', alpha), ('seed_init', seed)))
            for seed in args.seed_init
        ]
        value = mean([get_value(r) for r in rs])
        if value < obj_value:
            alpha_max = alpha
        else:
            alpha_min = alpha

        print(value, obj_value)


if __name__ == "__main__":
    main()
