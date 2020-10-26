# pylint: disable=missing-docstring, invalid-name, line-too-long
import argparse

from grid.exec import exec_blocking
from grid import load


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("log_dir", type=str)
    parser.add_argument("cmd", type=str)

    parser.add_argument("--alpha", type=float)
    parser.add_argument("--h", type=int)
    parser.add_argument("--wall", type=float)

    parser.add_argument("--seed_init", type=int, nargs='+')

    args = parser.parse_args()

    h = args.h

    while True:
        jammed = True
        for seed in args.seed_init:
            print('try h={}, seed={}'.format(h, seed))
            rs = load(args.log_dir, pred_args=lambda a: a.h == h and a.alpha == args.alpha and a.seed_init == seed)
            if any(r['regular']['dynamics'][-1]['train']['nd'] == 0 for r in rs):
                jammed = False
                break
            r = exec_blocking(args.log_dir, args.cmd, (('h', h), ('alpha', args.alpha), ('seed_init', seed), ('wall', args.wall)))
            if r['regular']['dynamics'][-1]['train']['nd'] == 0:
                jammed = False
                break

        if jammed:
            print('jammed!')
            break
        h -= 1


if __name__ == "__main__":
    main()
