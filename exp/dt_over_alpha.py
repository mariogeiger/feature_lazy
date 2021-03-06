import argparse
import math

from grid.exec import exec_one


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

    parser.add_argument("--nseed", type=int)

    parser.add_argument("--bs", type=int)

    args = parser.parse_args()

    runs = []

    for s in range(args.nseed):
        for i in range(args.i_start, args.i_end):
            for j in range(args.j_start, args.j_end):
                alpha = round_mantissa(args.base ** (i + j), args.tol)
                dt = round_mantissa(args.base ** (-i), args.tol)
                runs += [exec_one(args.log_dir, args.cmd, (('alpha', alpha), (args.tlike, dt), ('seed_init', s), ('bs', args.bs)))]

    for r in runs:
        r(False)  # wait but don't load


if __name__ == "__main__":
    main()
