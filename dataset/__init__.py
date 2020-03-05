# pylint: disable=no-member, E1102, C
"""
- Load mnist or cifar10
- perform PCA
- shuffle the dataset
- split in train and test set in an equilibrated way (same amount of each classes)
"""
import functools
from itertools import chain

import torch

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
        i = i.to(device=device, dtype=torch.long)
        outs += [(x, y, i)]

    return outs


def get_binary_dataset(dataset, ps, seeds, d, stretching, device=None, dtype=None):
    sets = get_normalized_dataset(dataset, ps, seeds, d, stretching)

    outs = []
    for x, y, i in sets:
        assert len(y.unique()) % 2 == 0
        x = x.to(device=device, dtype=dtype)
        y = (2 * (torch.arange(len(y)) % 2) - 1).type(x.dtype).to(device)
        i = i.to(device=device, dtype=torch.long)
        outs += [(x, y, i)]

    return outs


@functools.lru_cache(maxsize=2)
def get_normalized_dataset(dataset, ps, seeds, d=0, stretching=0):
    import torchvision

    transform = torchvision.transforms.ToTensor()

    torch.manual_seed(seeds[0])

    if dataset == "mnist":
        tr = torchvision.datasets.MNIST('~/.torchvision/datasets/MNIST', train=True, download=True, transform=transform)
        te = torchvision.datasets.MNIST('~/.torchvision/datasets/MNIST', train=False, transform=transform)
        x, y, i = dataset_to_tensors(list(tr) + list(te))
        x = center_normalize(x)
        return random_split(x, y, i, ps, seeds, y.unique())

    if dataset == "kmnist":
        tr = torchvision.datasets.KMNIST('~/.torchvision/datasets/KMNIST', train=True, download=True, transform=transform)
        te = torchvision.datasets.KMNIST('~/.torchvision/datasets/KMNIST', train=False, transform=transform)
        x, y, i = dataset_to_tensors(list(tr) + list(te))
        x = center_normalize(x)
        return random_split(x, y, i, ps, seeds, y.unique())

    if dataset == "emnist-letters":
        tr = torchvision.datasets.EMNIST('~/.torchvision/datasets/EMNIST', train=True, download=True, transform=transform, split='letters')
        te = torchvision.datasets.EMNIST('~/.torchvision/datasets/EMNIST', train=False, transform=transform, split='letters')
        x, y, i = dataset_to_tensors(list(tr) + list(te))
        x = center_normalize(x)
        return random_split(x, y, i, ps, seeds, y.unique())

    if dataset == "fashion":
        tr = torchvision.datasets.FashionMNIST('~/.torchvision/datasets/FashionMNIST', train=True, download=True, transform=transform)
        te = torchvision.datasets.FashionMNIST('~/.torchvision/datasets/FashionMNIST', train=False, transform=transform)
        x, y, i = dataset_to_tensors(list(tr) + list(te))
        x = center_normalize(x)
        return random_split(x, y, i, ps, seeds, y.unique())

    if dataset == "cifar10":
        tr = torchvision.datasets.CIFAR10('~/.torchvision/datasets/CIFAR10', train=True, download=True, transform=transform)
        te = torchvision.datasets.CIFAR10('~/.torchvision/datasets/CIFAR10', train=False, transform=transform)
        x, y, i = dataset_to_tensors(list(tr) + list(te))
        x = center_normalize(x)
        return random_split(x, y, i, ps, seeds, y.unique())

    if dataset == "cifar_catdog":
        tr = [(x, y) for x, y in torchvision.datasets.CIFAR10('~/.torchvision/datasets/CIFAR10', train=True, download=True, transform=transform) if y in [3, 5]]
        te = [(x, y) for x, y in torchvision.datasets.CIFAR10('~/.torchvision/datasets/CIFAR10', train=False, transform=transform) if y in [3, 5]]
        x, y, i = dataset_to_tensors(list(tr) + list(te))
        x = center_normalize(x)
        return random_split(x, y, i, ps, seeds, y.unique())

    if dataset == "cifar_shipbird":
        tr = [(x, y) for x, y in torchvision.datasets.CIFAR10('~/.torchvision/datasets/CIFAR10', train=True, download=True, transform=transform) if y in [8, 2]]
        te = [(x, y) for x, y in torchvision.datasets.CIFAR10('~/.torchvision/datasets/CIFAR10', train=False, transform=transform) if y in [8, 2]]
        x, y, i = dataset_to_tensors(list(tr) + list(te))
        x = center_normalize(x)
        return random_split(x, y, i, ps, seeds, y.unique())

    if dataset == "cifar_catplane":
        tr = [(x, y) for x, y in torchvision.datasets.CIFAR10('~/.torchvision/datasets/CIFAR10', train=True, download=True, transform=transform) if y in [3, 0]]
        te = [(x, y) for x, y in torchvision.datasets.CIFAR10('~/.torchvision/datasets/CIFAR10', train=False, transform=transform) if y in [3, 0]]
        x, y, i = dataset_to_tensors(list(tr) + list(te))
        x = center_normalize(x)
        return random_split(x, y, i, ps, seeds, y.unique())

    if dataset == "cifar_animal":
        tr = [(x, 0 if y in [0, 1, 8, 9] else 1) for x, y in torchvision.datasets.CIFAR10('~/.torchvision/datasets/CIFAR10', train=True, download=True, transform=transform)]
        te = [(x, 0 if y in [0, 1, 8, 9] else 1) for x, y in torchvision.datasets.CIFAR10('~/.torchvision/datasets/CIFAR10', train=False, transform=transform)]
        x, y, i = dataset_to_tensors(list(tr) + list(te))
        x = center_normalize(x)
        return random_split(x, y, i, ps, seeds, y.unique())

    if dataset == "catdog":
        tr = torchvision.datasets.ImageFolder('~/.torchvision/datasets/catdog', transform=transform)
        x, y, i = dataset_to_tensors(list(tr))
        x = center_normalize(x)
        return random_split(x, y, i, ps, seeds, y.unique())

    if dataset in ['stripe', 'sphere', 'cylinder', 'xnor']:
        out = []
        s = 0
        for p, seed in zip(ps, seeds):
            torch.manual_seed(seed + s)
            s += 1

            x = torch.randn(2 * p, d, dtype=torch.float64)
            if stretching:
                x = stretch(x, range(d//2, d), stretching)

            if dataset == 'stripe':
                y = (x[:, 0] > -0.3) * (x[:, 0] < 1.18549)
            if dataset == 'sphere':
                r = x.norm(dim=1)
                y = (r > d ** 0.5)
            if dataset == 'cylinder':
                dsph = d//2
                r = x[:,:dsph].norm(dim=1)
                y = (r > dsph ** 0.5)
            if dataset == 'xnor':
                threshold_x = 0.0
                threshold_y = 0.0
                y = (x[:, 0] > threshold_x) * (x[:, 1] > threshold_y) + (x[:, 0] < threshold_x) * (x[:, 1] < threshold_y)
            y = 2 * y - 1
            tr = [(x, y.item()) for x, y in zip(x, y)]
            x, y, _ = dataset_to_tensors(tr)
            out += [(x[:p], y[:p], torch.full_like(y, -1))]
        return out

    if dataset == 'sphere_grid':
        assert d == 2, "Spherical grid is only implemented in 2D"
        import math
        out = []
        s = 0
        for p, seed in zip(ps, seeds):
            torch.manual_seed(seed + s)
            s += 1

            # params, eventually to be added to parser
            bins = 500
            theta_bins = 50
            r_bins = bins // theta_bins

            n_per_box = p // bins

            assert p % bins == 0, f"p needs to be multiple of {bins}, number of bins"
            assert p % theta_bins == 0, f"p needs to be multiple of {theta_bins}, number of angular bins"

            rgauss = inverf2(torch.arange(r_bins).double().div_(r_bins))
            # cutting the last bin of the gaussian
            rgauss = torch.cat((rgauss, torch.ones(1)*4.0))
            rdiff = rgauss[1:] - rgauss[:-1]
            x = torch.zeros(p, d)

            for i in range(bins):
                theta = (torch.rand(n_per_box) + (i % theta_bins)) / theta_bins * 2 * math.pi
                r = torch.rand(n_per_box) * rdiff[i // theta_bins] + rgauss[i // theta_bins]
                x[i * n_per_box:(i + 1) * n_per_box, 0] = r.mul(theta.cos())
                x[i * n_per_box:(i + 1) * n_per_box, 1] = r.mul(theta.sin())

            r = x.norm(dim=1)
            y = 2 * (r > rgauss[len(rgauss) // 2]) - 1
            tr = [(x, y.item()) for x, y in zip(x, y)]
            x, y, _ = dataset_to_tensors(tr)
            out += [(x, y, torch.full_like(y, -1))]
        return out

    raise ValueError("unknown dataset")


def dataset_to_tensors(dataset):
    dataset = [(x.type(torch.float64), int(y), i) for i, (x, y) in enumerate(dataset)]
    classes = sorted({y for x, y, i in dataset})

    sets = [[(x, y, i) for x, y, i in dataset if y == z] for z in classes]

    sets = [
        [x[i] for i in torch.randperm(len(x))]
        for x in sets
    ]

    dataset = list(chain(*zip(*sets)))

    x = torch.stack([x for x, y, i in dataset])
    y = torch.tensor([y for x, y, i in dataset], dtype=torch.long)
    i = torch.tensor([i for x, y, i in dataset], dtype=torch.long)
    return x, y, i


def center_normalize(x):
    x = x - x.mean(0)
    x = (x[0].numel() ** 0.5) * x / x.flatten(1).norm(dim=1).view(-1, *(1,) * (x.dim() - 1))
    return x


def random_split(x, y, i, ps, seeds, classes):
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
    return [(x[:p], y[:p], i[:p])] + random_split(x[p:], y[p:], i[p:], ps, seeds, classes)
