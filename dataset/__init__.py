# pylint: disable=no-member, E1102, C
"""
- Load mnist or cifar10
- perform PCA
- shuffle the dataset
- split in train and test set in an equilibrated way (same amount of each classes)
"""
import functools
import math
from itertools import chain

import math
import scipy.special
import torch
import torch.nn.functional as F

def inverf2(x):
    """ Inverse error function in 2d."""
    return (-2 * (1 - x).log()).sqrt()


def stretch(x, dims, factor):
    """Stretch specific dimensions - dims - by factor."""
    x[:,dims] *= factor
    return x


def pca(x, d, whitening):
    """
    :param x: [P, ...]
    :return: [P, d]
    """

    z = x.flatten(1)
    mu = z.mean(0)
    cov = (z - mu).t() @ (z - mu) / len(z)

    val, vec = cov.symeig(eigenvectors=True)
    val, idx = val.sort(descending=True)
    vec = vec[:, idx]

    u = (z - mu) @ vec[:, :d]
    if whitening:
        u.mul_(val[:d].rsqrt())
    else:
        u.mul_(val[:d].mean().rsqrt())

    return u


# def get_binary_pca_dataset(dataset, p, d, whitening, seed=None, device=None):
#     if seed is None:
#         seed = torch.randint(2 ** 32, (), dtype=torch.long).item()

#     x, y = get_normalized_dataset(dataset, seed)

#     x = pca(x, d, whitening).to(device)
#     y = (2 * (torch.arange(len(y)) % 2) - 1).type(x.dtype).to(device)

#     xtr = x[:p]
#     xte = x[p:]
#     ytr = y[:p]
#     yte = y[p:]

#     return (xtr, ytr), (xte, yte)


def get_dataset(dataset, ps, seeds, d, device=None, dtype=None):
    sets = get_normalized_dataset(dataset, ps, seeds, d)

    outs = []
    for x, y, i in sets:
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=torch.long)
        outs += [(x, y, i)]

    return outs


def get_binary_dataset(dataset, ps, seeds, d, stretching, params, device=None, dtype=None):
    sets = get_normalized_dataset(dataset, ps, seeds, d, stretching, params)

    outs = []
    for x, y, i in sets:
        x = x.to(device=device, dtype=dtype)

        assert len(y.unique()) % 2 == 0
        b = x.new_zeros(len(y))
        for j, z in enumerate(y.unique()):
            if j % 2 == 0:
                b[y == z] = 1
            else:
                b[y == z] = -1

        outs += [(x, b, i)]

    return outs


@functools.lru_cache(maxsize=2)
def get_normalized_dataset(dataset, ps, seeds, d=0, stretching=0, params=None):
    import torchvision

    transform = torchvision.transforms.ToTensor()

    torch.manual_seed(seeds[0])

    if dataset == "mnist":
        tr = torchvision.datasets.MNIST('~/.torchvision/datasets/MNIST', train=True, download=True, transform=transform)
        te = torchvision.datasets.MNIST('~/.torchvision/datasets/MNIST', train=False, transform=transform)
        x, y, i = intertwine_labels(*dataset_to_tensors(list(tr) + list(te)))
        x = center_normalize(x)
        return intertwine_split(x, y, i, ps, seeds, y.unique())

    if dataset == "kmnist":
        tr = torchvision.datasets.KMNIST('~/.torchvision/datasets/KMNIST', train=True, download=True, transform=transform)
        te = torchvision.datasets.KMNIST('~/.torchvision/datasets/KMNIST', train=False, transform=transform)
        x, y, i = intertwine_labels(*dataset_to_tensors(list(tr) + list(te)))
        x = center_normalize(x)
        return intertwine_split(x, y, i, ps, seeds, y.unique())

    if dataset == "emnist-letters":
        tr = torchvision.datasets.EMNIST('~/.torchvision/datasets/EMNIST', train=True, download=True, transform=transform, split='letters')
        te = torchvision.datasets.EMNIST('~/.torchvision/datasets/EMNIST', train=False, transform=transform, split='letters')
        x, y, i = intertwine_labels(*dataset_to_tensors(list(tr) + list(te)))
        x = center_normalize(x)
        return intertwine_split(x, y, i, ps, seeds, y.unique())

    if dataset == "fashion":
        tr = torchvision.datasets.FashionMNIST('~/.torchvision/datasets/FashionMNIST', train=True, download=True, transform=transform)
        te = torchvision.datasets.FashionMNIST('~/.torchvision/datasets/FashionMNIST', train=False, transform=transform)
        x, y, i = intertwine_labels(*dataset_to_tensors(list(tr) + list(te)))
        x = center_normalize(x)
        return intertwine_split(x, y, i, ps, seeds, y.unique())

    if dataset == "cifar10":
        tr = torchvision.datasets.CIFAR10('~/.torchvision/datasets/CIFAR10', train=True, download=True, transform=transform)
        te = torchvision.datasets.CIFAR10('~/.torchvision/datasets/CIFAR10', train=False, transform=transform)
        x, y, i = intertwine_labels(*dataset_to_tensors(list(tr) + list(te)))
        x = center_normalize(x)
        return intertwine_split(x, y, i, ps, seeds, y.unique())

    if dataset == "cifar_catdog":
        tr = [(x, y) for x, y in torchvision.datasets.CIFAR10('~/.torchvision/datasets/CIFAR10', train=True, download=True, transform=transform) if y in [3, 5]]
        te = [(x, y) for x, y in torchvision.datasets.CIFAR10('~/.torchvision/datasets/CIFAR10', train=False, transform=transform) if y in [3, 5]]
        x, y, i = intertwine_labels(*dataset_to_tensors(list(tr) + list(te)))
        x = center_normalize(x)
        return intertwine_split(x, y, i, ps, seeds, y.unique())

    if dataset == "cifar_shipbird":
        tr = [(x, y) for x, y in torchvision.datasets.CIFAR10('~/.torchvision/datasets/CIFAR10', train=True, download=True, transform=transform) if y in [8, 2]]
        te = [(x, y) for x, y in torchvision.datasets.CIFAR10('~/.torchvision/datasets/CIFAR10', train=False, transform=transform) if y in [8, 2]]
        x, y, i = intertwine_labels(*dataset_to_tensors(list(tr) + list(te)))
        x = center_normalize(x)
        return intertwine_split(x, y, i, ps, seeds, y.unique())

    if dataset == "cifar_catplane":
        tr = [(x, y) for x, y in torchvision.datasets.CIFAR10('~/.torchvision/datasets/CIFAR10', train=True, download=True, transform=transform) if y in [3, 0]]
        te = [(x, y) for x, y in torchvision.datasets.CIFAR10('~/.torchvision/datasets/CIFAR10', train=False, transform=transform) if y in [3, 0]]
        x, y, i = intertwine_labels(*dataset_to_tensors(list(tr) + list(te)))
        x = center_normalize(x)
        return intertwine_split(x, y, i, ps, seeds, y.unique())

    if dataset == "cifar_animal":
        tr = [(x, 0 if y in [0, 1, 8, 9] else 1) for x, y in torchvision.datasets.CIFAR10('~/.torchvision/datasets/CIFAR10', train=True, download=True, transform=transform)]
        te = [(x, 0 if y in [0, 1, 8, 9] else 1) for x, y in torchvision.datasets.CIFAR10('~/.torchvision/datasets/CIFAR10', train=False, transform=transform)]
        x, y, i = intertwine_labels(*dataset_to_tensors(list(tr) + list(te)))
        x = center_normalize(x)
        return intertwine_split(x, y, i, ps, seeds, y.unique())

    if dataset == "catdog":
        tr = torchvision.datasets.ImageFolder('~/.torchvision/datasets/catdog', transform=transform)
        x, y, i = intertwine_labels(*dataset_to_tensors(list(tr)))
        x = center_normalize(x)
        return intertwine_split(x, y, i, ps, seeds, y.unique())

    out = []
    s = 0
    for p, seed in zip(ps, seeds):
        s += seed + 1
        torch.manual_seed(s)
        x = torch.randn(p, d, dtype=torch.float64)

        if stretching:
            ds = d // 2 if params[0] is None else params[1]
            x = stretch(x, range(ds, d), stretching)

        if dataset == 'stripe':
            y = (x[:, 0] > -0.3) * (x[:, 0] < 1.18549)
        if dataset == 'sphere':
            r = x.norm(dim=1)
            y = (r**2 > d - 2 / 3)
        if dataset == 'cylinder':
            dsph = d // 2 if params[0] is None else params[0]
            r = x[:, :dsph].norm(dim=1)
            y = (r**2 > dsph  - 2 / 3)
        if dataset == 'cube':
            a = scipy.special.erfinv(0.5**(1 / d)) * 2**0.5
            y = (x.abs() < a).all(1)
        if dataset == 'xnor':
            y = (x[:, 0] > 0) * (x[:, 1] > 0) + (x[:, 0] < 0) * (x[:, 1] < 0)
        if dataset == 'and':  # classical AND logic gate (only two relevant dimensions, x1 and x2, no matter what the input dimension d is)
            y = (x[:, 0] > 0) * (x[:, 1] > 0)
        if dataset == 'andD':  # multi-dimensional AND logic gate (all d dimensions are relevant)
            y = (x > 0).all(1)
        if dataset == 'sphere_grid':
            assert d == 2, "Spherical grid is only implemented in 2D"

            bins = 500 if params[0] is None else params[0]
            theta_bins = 50 if params[1] is None else params[1]
            assert p % bins == 0, f"p needs to be multiple of {bins}, number of bins"
            assert p % theta_bins == 0, f"p needs to be multiple of {theta_bins}, number of angular bins"
            r_bins = bins // theta_bins
            ppc = p // bins # points per cell

            r_spacing = inverf2(torch.arange(r_bins).double().div_(r_bins))
            # cutting the last bin of the gaussian which would go to infinity
            infty = 4.0
            r_spacing = torch.cat((r_spacing, torch.ones(1) * infty))

            r_diff = r_spacing[1:] - r_spacing[:-1]
            x = torch.zeros(p, d)

            for i in range(bins):
                theta = (torch.rand(ppc) + (i % theta_bins)) / theta_bins * 2 * math.pi
                r = torch.rand(ppc) * r_diff[i // theta_bins] + r_spacing[i // theta_bins]
                x[i * ppc:(i + 1) * ppc, 0] = r.mul(theta.cos())
                x[i * ppc:(i + 1) * ppc, 1] = r.mul(theta.sin())
            r = x.norm(dim=1)
            y = 2 * (r > r_spacing[len(r_spacing) // 2]) - 1
        if dataset == 'signal_1d':
            n0 = 2 if params[0] is None else params[0]
            C0 = .64 if params[1] is None else params[1]

            x = torch.zeros(p, d)
            y = torch.zeros(p)
            c = torch.randn(p, 2, n0)
            psi = torch.linspace(0, math.pi, d).cos().reshape(1, 1, -1) / d

            for pi in range(p):
                for k in range(n0):
                    x[pi] += c[pi, 0, k] * torch.linspace(0, (k + 1) * math.pi, d).cos() + \
                             c[pi, 1, k] * torch.linspace(0, (k + 1) * math.pi, d).sin()
                y[pi] += F.conv1d(torch.cat((x[pi], x[pi, :-1]), dim=0).reshape(1, 1, -1), psi).max(dim=2).values[0].item() - C0
            y = 2 * (y > 0) - 1
        y = y.to(dtype=torch.long)
        out += [(x, y, None)]
    return out


def dataset_to_tensors(dataset):
    dataset = [(x.type(torch.float64), int(y), i) for i, (x, y) in enumerate(dataset)]
    x = torch.stack([x for x, y, i in dataset])
    y = torch.tensor([y for x, y, i in dataset], dtype=torch.long)
    i = torch.tensor([i for x, y, i in dataset], dtype=torch.long)
    return x, y, i


def intertwine_labels(x, y, i):
    classes = y.unique()
    sets = [(x[y == z], y[y == z], i[y == z]) for z in classes]

    del x, y, i

    sets = [
        (x[rp], y[rp], i[rp])
        for x, y, i, rp in
        ((x, y, i, torch.randperm(len(x))) for x, y, i in sets)
    ]

    x = torch.stack(list(chain(*zip(*(x for x, y, i in sets)))))
    y = torch.stack(list(chain(*zip(*(y for x, y, i in sets)))))
    i = torch.stack(list(chain(*zip(*(i for x, y, i in sets)))))
    return x, y, i


def center_normalize(x):
    x = x - x.mean(0)
    x = (x[0].numel() ** 0.5) * x / x.flatten(1).norm(dim=1).view(-1, *(1,) * (x.dim() - 1))
    return x


def intertwine_split(x, y, i, ps, seeds, classes):
    assert len(ps) == len(seeds)

    if len(ps) == 0:
        return []

    xs = [(x[y == z], i[y == z]) for z in classes]

    ps = list(ps)
    seeds = list(seeds)

    p = ps.pop(0)
    seed = seeds.pop(0)

    torch.manual_seed(seed)
    xx = []
    ii = []
    for x, i in xs:
        rp = torch.randperm(len(x))
        xx.append(x[rp])
        ii.append(i[rp])

    ys = [torch.full((len(x),), z, dtype=torch.long) for x, z in zip(xx, classes)]

    x = torch.stack(list(chain(*zip(*xx))))
    y = torch.stack(list(chain(*zip(*ys))))
    i = torch.stack(list(chain(*zip(*ii))))

    assert len(x) >= p
    return [(x[:p], y[:p], i[:p])] + intertwine_split(x[p:], y[p:], i[p:], ps, seeds, classes)
