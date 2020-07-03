import numpy as np
import scipy.spatial as sps

def h2(p):
    """Binary entropy function."""
    eps = 1e-10
    return - p * np.log2(p + eps) - (1-p) * np.log2(1-p + eps)

def cond_proba(tree, point, y, k=3):
    """ Query a cKDTree for kNNs for y>0 and y<=0.
        point is where the query is centered.
        Returns conditional probability of the label > 0:
                    P(y>0 | x) = q+ = 1 / (1 + (r+/r-)^d)"""

    eps = 1e-10
    d = point.shape[0]

    flag_p = False
    flag_m = False

    k_large = k
    while not (flag_p * flag_m):
        k_large = min(k_large*2 + 5, len(y) - 1)

        q = tree.query(point, k_large, p=2)

        if sum(y[q[1]] > 0) >= k and not flag_p:
            i = np.where(y[q[1]] > 0)[0][k-1]
            rp = q[0][i] + eps
            flag_p = True

        if sum(y[q[1]] <= 0) >= k and not flag_m:
            i = np.where(y[q[1]] <= 0)[0][k-1]
            rm = q[0][i] + eps
            flag_m = True

    qp = 1 / (1 + (rp/rm)**d)

    if qp == 0:
        qp = eps

    assert qp > 0, "Set probability larger than zero"
    assert qp <= 1, "Set probability smaller than one"

    return qp

def mi_binary(x, y, k=3):
    """Compute mutual information for continuous variable x,
    and binary variable y with p(y=+) = p(y=-) = 0.5 ."""

    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"

    eps = 1e-10  # small noise to break degeneracy.
    x += eps * np.random.rand(*x.shape)

    tree = sps.cKDTree(x)
    q = np.asarray([cond_proba(tree, point, y, k=k) for point in x])

    return 1 - np.mean(h2(q))


def component_mi(eigenvectors, y, k=5, max_r=10, standardize=True):
    """Compute mutual information between last max_r elements of eigenvectors
    and y.
    Returns a list [I(phi_r; y)] for i = 1 to max_r .
    (input eigenvectors normally a torch.tensor)"""
    mi_ = []

    for r in range(1, max_r + 1):
        eig = eigenvectors[:, -r]

        if standardize:
            eig = (eig - eig.mean()) / eig.std()

        mi_.append(mi_binary(np.asarray([[e] for e in eig]), y, k=k))

    return mi_

