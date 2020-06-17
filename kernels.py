# pylint: disable=no-member, C, not-callable
"""
Computes the Gram matrix of a given model
"""


def compute_kernels(f, xtr, xte, parameters=None):
    from hessian import gradient

    if parameters is None:
        parameters = list(f.parameters())

    ktrtr = xtr.new_zeros(len(xtr), len(xtr))
    ktetr = xtr.new_zeros(len(xte), len(xtr))
    ktete = xtr.new_zeros(len(xte), len(xte))

    params = []
    current = []
    for p in sorted(parameters, key=lambda p: p.numel(), reverse=True):
        current.append(p)
        if sum(p.numel() for p in current) > 2e9 // (8 * (len(xtr) + len(xte))):
            if len(current) > 1:
                params.append(current[:-1])
                current = current[-1:]
            else:
                params.append(current)
                current = []
    if len(current) > 0:
        params.append(current)

    for i, p in enumerate(params):
        print("[{}/{}] [len={} numel={}]".format(i, len(params), len(p), sum(x.numel() for x in p)), flush=True)

        jtr = xtr.new_empty(len(xtr), sum(u.numel() for u in p))  # (P, N~)
        jte = xte.new_empty(len(xte), sum(u.numel() for u in p))  # (P, N~)

        for j, x in enumerate(xtr):
            jtr[j] = gradient(f(x[None]), p)  # (N~)

        for j, x in enumerate(xte):
            jte[j] = gradient(f(x[None]), p)  # (N~)

        ktrtr.add_(jtr @ jtr.t())
        ktetr.add_(jte @ jtr.t())
        ktete.add_(jte @ jte.t())
        del jtr, jte

    return ktrtr, ktetr, ktete


def twonn_ratio(dist):
    """
    dist: (N, N)-array of the euclidean distances between each pair of the
    population of N points.
    """
    dist = dist.sort(1).values
    assert dist[:, 0].norm() == 0, dist[:, 0]
    return dist[:, 2] / dist[:, 1]


def intrinsic_dimension(mu):
    """
    mu: array of the ratios between second and first neighbours.
    """

    n = len(mu)
    v = mu.log().sum()

    # intrinsic dimension
    d = (n + 1) / v
    # std deviation around id
    sigma = (n - 1)**0.5 / v

    return d, sigma


def kernel_intdim(k):
    dist = (k.diag().reshape(-1, 1) + k.diag().reshape(1, -1) - 2 * k).sqrt()

    mu = twonn_ratio(dist)
    d, sigma = intrinsic_dimension(mu)

    return d, sigma
