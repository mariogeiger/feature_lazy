# pylint: disable=no-member, invalid-name, missing-docstring, not-callable
import itertools
import random

import torch


def gen(length):
    patterns = [
        torch.tensor([0.0, 1, -1, 1]),

        torch.tensor([0.0]),

        torch.tensor([0.0, 1]),
        torch.tensor([0.0, 1, 1]),
        torch.tensor([0.0, 1, 1, 1]),
        torch.tensor([0.0, 1, 1, -1]),

        torch.tensor([0.0, 1, -1]),
        torch.tensor([0.0, 1, -1, -1]),
    ]

    xs = random.choices(patterns, [0.3 ** len(x) for x in patterns], k=length + 1)
    xs = xs[:next(i for i, n in enumerate(itertools.accumulate(len(x) for x in xs)) if n > length)]

    y = sum(p is patterns[0] for p in xs)
    x = torch.cat([x * random.choice([-1, 1]) for x in xs])
    x = torch.cat([x, torch.zeros(length - len(x))])

    return x, y
