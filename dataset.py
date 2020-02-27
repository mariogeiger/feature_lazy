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


def pca(x, d, whitening):
    '''
    :param x: [P, ...]
    :return: [P, d]
    '''

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


def get_dataset(dataset, p, d, seed=None, device=None):
    if seed is None:
        seed = torch.randint(2 ** 32, (), dtype=torch.long).item()

    x, y = get_normalized_dataset(dataset, p, d, seed)

    x = x.to(device)
    y = y.to(device)

    return x, y


def get_binary_dataset(dataset, p, d, seed=None, device=None):
    if seed is None:
        seed = torch.randint(2 ** 32, (), dtype=torch.long).item()

    x, y = get_normalized_dataset(dataset, p, d, seed)

    x = x.to(device)
    y = (2 * (torch.arange(len(y)) % 2) - 1).type(x.dtype).to(device)

    return x, y


@functools.lru_cache(maxsize=2)
def get_normalized_dataset(dataset, p=0, d=0, seed=0):
    import torchvision

    torch.manual_seed(seed)

    transform = torchvision.transforms.ToTensor()

    if dataset == "mnist":
        tr = torchvision.datasets.MNIST('~/.torchvision/datasets/MNIST', train=True, download=True, transform=transform)
        te = torchvision.datasets.MNIST('~/.torchvision/datasets/MNIST', train=False, transform=transform)
        x, y = dataset_to_tensors(list(tr) + list(te))
        x = center_normalize(x)
    elif dataset == "kmnist":
        tr = torchvision.datasets.KMNIST('~/.torchvision/datasets/KMNIST', train=True, download=True, transform=transform)
        te = torchvision.datasets.KMNIST('~/.torchvision/datasets/KMNIST', train=False, transform=transform)
        x, y = dataset_to_tensors(list(tr) + list(te))
        x = center_normalize(x)
    elif dataset == "emnist-letters":
        tr = torchvision.datasets.EMNIST('~/.torchvision/datasets/EMNIST', train=True, download=True, transform=transform, split='letters')
        te = torchvision.datasets.EMNIST('~/.torchvision/datasets/EMNIST', train=False, transform=transform, split='letters')
        x, y = dataset_to_tensors(list(tr) + list(te))
        x = center_normalize(x)
    elif dataset == "fashion":
        tr = torchvision.datasets.FashionMNIST('~/.torchvision/datasets/FashionMNIST', train=True, download=True, transform=transform)
        te = torchvision.datasets.FashionMNIST('~/.torchvision/datasets/FashionMNIST', train=False, transform=transform)
        x, y = dataset_to_tensors(list(tr) + list(te))
        x = center_normalize(x)
    elif dataset == "cifar10":
        tr = torchvision.datasets.CIFAR10('~/.torchvision/datasets/CIFAR10', train=True, download=True, transform=transform)
        te = torchvision.datasets.CIFAR10('~/.torchvision/datasets/CIFAR10', train=False, transform=transform)
        x, y = dataset_to_tensors(list(tr) + list(te))
        x = center_normalize(x)
    elif dataset == "cifar_catdog":
        tr = [(x, y) for x, y in torchvision.datasets.CIFAR10('~/.torchvision/datasets/CIFAR10', train=True, download=True, transform=transform) if y in [3, 5]]
        te = [(x, y) for x, y in torchvision.datasets.CIFAR10('~/.torchvision/datasets/CIFAR10', train=False, transform=transform) if y in [3, 5]]
        x, y = dataset_to_tensors(list(tr) + list(te))
        x = center_normalize(x)
    elif dataset == "cifar_shipbird":
        tr = [(x, y) for x, y in torchvision.datasets.CIFAR10('~/.torchvision/datasets/CIFAR10', train=True, download=True, transform=transform) if y in [8, 2]]
        te = [(x, y) for x, y in torchvision.datasets.CIFAR10('~/.torchvision/datasets/CIFAR10', train=False, transform=transform) if y in [8, 2]]
        x, y = dataset_to_tensors(list(tr) + list(te))
        x = center_normalize(x)
    elif dataset == "cifar_catplane":
        tr = [(x, y) for x, y in torchvision.datasets.CIFAR10('~/.torchvision/datasets/CIFAR10', train=True, download=True, transform=transform) if y in [3, 0]]
        te = [(x, y) for x, y in torchvision.datasets.CIFAR10('~/.torchvision/datasets/CIFAR10', train=False, transform=transform) if y in [3, 0]]
        x, y = dataset_to_tensors(list(tr) + list(te))
        x = center_normalize(x)
    elif dataset == "cifar_animal":
        tr = [(x, 0 if y in [0, 1, 8, 9] else 1) for x, y in torchvision.datasets.CIFAR10('~/.torchvision/datasets/CIFAR10', train=True, download=True, transform=transform)]
        te = [(x, 0 if y in [0, 1, 8, 9] else 1) for x, y in torchvision.datasets.CIFAR10('~/.torchvision/datasets/CIFAR10', train=False, transform=transform)]
        x, y = dataset_to_tensors(list(tr) + list(te))
        x = center_normalize(x)
    elif dataset == "catdog":
        tr = torchvision.datasets.ImageFolder('~/.torchvision/datasets/catdog', transform=transform)
        x, y = dataset_to_tensors(list(tr))
        x = center_normalize(x)
    elif dataset in ['stripe', 'sphere' , 'xnor']:
        x = torch.randn(2 * p, d, dtype=torch.float64)
        if dataset == 'stripe':
            y = (x[:, 0] > -0.3) * (x[:, 0] < 1.18549)
        if dataset == 'sphere':
            r = x.norm(dim=1)
            y = (r > d**0.5)
        if dataset == 'xnor':
            threshold_x = 0.0 ; threshold_y = 0.0 
            y = (x[:, 0] > threshold_x) * (x[:, 1] > threshold_y) + (x[:, 0] < threshold_x) * (x[:, 1] < threshold_y)
        y = 2 * y - 1
        tr = [(x, y.item()) for x, y in zip(x, y)]
        x, y = dataset_to_tensors(tr)
    elif dataset == "pat1d":
        from pat1d import gen
        tr = []
        while len(tr) < 2 * p:
            x, y = gen(70)
            tr.append((x.view(1, -1), y > 0))
        x, y = dataset_to_tensors(tr)
    else:
        raise ValueError("unknown dataset")

    if p > 0:
        assert len(x) >= p, (x.shape, p)
        x, y = x[:p], y[:p]
    if d is not None and d > 0:
        assert x.flatten(1).shape[1] == d
    return x, y


def dataset_to_tensors(dataset):
    dataset = [(x.type(torch.float64), int(y)) for x, y in dataset]
    classes = sorted({y for x, y in dataset})

    sets = [[(x, y) for x, y in dataset if y == i] for i in classes]

    sets = [
        [x[i] for i in torch.randperm(len(x))]
        for x in sets
    ]

    dataset = list(chain(*zip(*sets)))

    x = torch.stack([x for x, y in dataset])
    y = torch.tensor([y for x, y in dataset], dtype=torch.long)
    return x, y


def center_normalize(x):
    x = x - x.mean(0)
    x = (x[0].numel() ** 0.5) * x / x.flatten(1).norm(dim=1).view(-1, *(1,) * (x.dim() - 1))
    return x
