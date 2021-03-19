import argparse
import math

from grid.exec import exec_blocking


def round_mantissa(x, n):
    m = round(math.log2(x))
    return round(2**(n-m) * x) * 2**(m-n)


def mean(z):
    return sum(z) / len(z)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("log_dir", type=str)
    parser.add_argument("cmd", type=str)

    parser.add_argument("--tlike", type=str)

    parser.add_argument("--i_start", type=int)
    parser.add_argument("--i_end", type=int)

    parser.add_argument("--j_start", type=int)
    parser.add_argument("--j_end", type=int)

    parser.add_argument("--base", type=float)
    parser.add_argument("--tol", type=int, default=5)

    args = parser.parse_args()

    runs = []

    for i in range(args.i_start, args.i_end):
        for j in range(args.j_start, args.j_end):
            alpha = round_mantissa(args.base, i + j, args.tol)
            dt = round_mantissa(args.base, -i, args.tol)
            runs += [exec_blocking(args.log_dir, args.cmd, ('alpha', alpha), (args.tlike, dt))]

    for r in runs:
        r(False)  # wait but don't load


if __name__ == "__main__":
    main()
