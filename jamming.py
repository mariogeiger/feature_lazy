# pylint: disable=missing-docstring, invalid-name
import argparse

from grid.exec import exec_blocking


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("log_dir", type=str)
    parser.add_argument("cmd", type=str)

    parser.add_argument("--alpha", type=float)
    parser.add_argument("--h", type=int)

    parser.add_argument("--seed_init", type=int)
    parser.add_argument("--ptr", type=int)

    args = parser.parse_args()

    h = args.h

    while True:
        param = (('h', h), ('alpha', args.alpha), ('seed_init', args.seed_init), ('ptr', args.ptr))
        data = exec_blocking(args.log_dir, args.cmd, param)
        dyn = data['regular']['dynamics']
        if dyn[-1]['train']['nd'] > 0:
            break
        h -= 1


if __name__ == "__main__":
    main()
