# pylint: disable=missing-docstring, invalid-name
import argparse
import math

from grid.exec import exec_blocking


def round_mantissa(x, n):
    m = round(math.log2(x))
    return round(2**(n-m) * x) * 2**(m-n)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("log_dir", type=str)
    parser.add_argument("cmd", type=str)

    parser.add_argument("--alpha_min", type=float)
    parser.add_argument("--alpha_max", type=float)
    parser.add_argument("--tol", type=int)
    parser.add_argument("--c", type=float)
    parser.add_argument("--h", type=int)

    parser.add_argument("--seed_init", type=int)
    parser.add_argument("--ptr", type=int)

    args = parser.parse_args()

    alpha_min = args.alpha_min
    alpha_max = args.alpha_max

    while True:
        alpha = math.sqrt(alpha_min * alpha_max)
        alpha = round_mantissa(alpha, args.tol)

        if alpha in [alpha_min, alpha_max]:
            break

        print("{} < {} < {}".format(alpha_min, alpha, alpha_max))

        param = (('h', args.h), ('alpha', alpha), ('seed_init', args.seed_init), ('ptr', args.ptr))
        data = exec_blocking(args.log_dir, args.cmd, param)
        if data['delta_kernel']['traink'] < args.c * data['delta_kernel']['init']['traink']['norm']:
            alpha_max = alpha
        else:
            alpha_min = alpha

        print(data['delta_kernel']['traink'] / data['delta_kernel']['init']['traink']['norm'])


if __name__ == "__main__":
    main()
