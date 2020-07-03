import os
import argparse
import numpy as np
import torch
from scipy.integrate import quad
from scipy.special import erf, erfc

def label(x, xmin, xmax):
    if xmin < x < xmax:
        return -1.
    else:
        return 1.

def f_Si(w1, z, xtr, ytr, i):
    return np.sum(xtr[:, i] * ytr * np.heaviside(w1 * (xtr[:, 0] - z), 0)) / len(ytr)

def get_neuron(r, i):
    neuron = {
        "t": list(),
        "w": list(),
        "b": list(),
        "beta": list(),
        "z": list()
    }
    for x in r["regular"]["dynamics"]:
        neuron["t"].append(x["t"])
        neuron["beta"].append(x["neurons"]["beta"][i])
        neuron["b"].append(x["neurons"]["b"][i])
        neuron["w"].append([x["neurons"]["w"][i][j] for j in range(r["args"].d)])
        w2 = np.sum([neuron["w"][-1][j]**2 for j in range(r["args"].d)])
        neuron["z"].append([-r["args"].d**0.5 * neuron["b"][-1] * neuron["w"][-1][j] / w2 for j in range(r["args"].d)])

    return {key: np.array(x) for key, x in neuron.items()}

def get_list_neurons_alive(r, limit=1e2):
    idx = list()
    for i in range(r["args"].save_neurons):
        if np.abs(r["regular"]["dynamics"][-1]["neurons"]["beta"][i]) > limit:
            idx.append(i)

    return idx

def get_list_zstar(r, zstar):
    idx = list()
    for i in range(r["args"].save_neurons):
        neuron = get_neuron(r, i)
        istar = get_istar(neuron, tstar)
        if np.abs(neuron["z1"][istar] - zstar) < 1e-1:
            idx.append(i)
    return idx

def get_istar(neuron, tstar):
    for i, t in enumerate(neuron["t"]):
        if t > tstar:
            return i

def first_moment1(y, d, xmin, xmax, lim):
    omega = y[:d]
    b, beta = y[d:]
    omega_perp2 = np.sum([omega[i]**2 for i in range(1, d)])
    integ = lambda x: (2*np.pi)**-0.5*np.exp(-x**2/2) * label(x, xmin, xmax) * x * 0.5*erfc(-(b*d**0.5 + omega[0]*x)/(2*omega_perp2)**0.5)
    return quad(integ, -lim, lim)[0]

def first_momenti(y, i, d, xmin, xmax, lim):
    omega = y[:d]
    b, beta = y[d:]
    omega_perp2 = np.sum([omega[i]**2 for i in range(1, d)])
    omega2 = omega[0]**2 + omega_perp2
    integ = lambda x: (2*np.pi)**-0.5*np.exp(-x**2/2) * label((omega_perp2/omega2)**0.5*x - d**0.5*b*omega[0]/omega2, xmin, xmax)
    return (2*np.pi)**-0.5*omega[i]/omega2**0.5 * np.exp(-d*b**2/(2*omega2)) * quad(integ, -lim, lim)[0]

def first_momentb(y, d, xmin, xmax, lim):
    omega = y[:d]
    b, beta = y[d:]
    omega_perp2 = np.sum([omega[i]**2 for i in range(1, d)])
    integ = lambda x: (2*np.pi)**-0.5*np.exp(-x**2/2) * label(x, xmin, xmax) * 0.5*erfc(-(b*d**0.5 + omega[0]*x)/(2*omega_perp2)**0.5)
    return quad(integ, -lim, lim)[0]

def second_moment1(y, d, lim):
    omega = y[:d]
    b, beta = y[d:]
    omega_perp2 = np.sum([omega[i]**2 for i in range(1, d)])
    integ = lambda x: (2*np.pi)**-0.5*np.exp(-x**2/2) * 0.5*erfc(-(b*d**0.5 + omega[0]*x)/(2*omega_perp2)**0.5)
    return quad(integ, -lim, lim)[0]

def second_momenti(y, i, d, lim):
    omega = y[:d]
    b, beta = y[d:]
    omega_perp2 = np.sum([omega[j]**2 for j in range(d) if i != j])
    integ = lambda x: (2*np.pi)**-0.5*np.exp(-x**2/2) * 0.5*erfc(-(b*d**0.5 + omega[i]*x)/(2*omega_perp2)**0.5)
    return quad(integ, -lim, lim)[0]

def second_momentb(y, d, lim):
    omega = y[:d]
    b, beta = y[d:]
    omega2 = np.sum([omega[i]**2 for i in range(d)])
    return 0.5*erfc(-(b*d**0.5)/(2*omega2)**0.5)

def model(y0, dt, d, a, T, xmin, xmax, p=-1, Ni=None, lim=np.infty):

    dt /= a
    T /= a

    c0 = np.sum([y0[j]**2 for j in range(d)]) + y0[d]**2 - y0[d+1]**2
    y = [y0]
    t = [0]
    while True:
        omega = y[-1][:d]
        b, beta = y[-1][d:]

        mu1 = first_moment1(y[-1], d, xmin, xmax, lim)
        mui = [first_momenti(y[-1], i, d, xmin, xmax, lim) for i in range(1, d)]
        mub = first_momentb(y[-1], d, xmin, xmax, lim)

        s1 = a * beta * mu1
        si = [a * beta * _mui for _mui in mui]
        sb = a * d**0.5 * beta * mub

        if p > 0:
            sigmai = [(second_momenti(y[-1], i, d, lim) - mui[i-1]**2)**0.5 for i in range(1, d)]
            for i in range(1, d):
                si[i-1] += a * beta * sigmai[i-1] / p**0.5 * Ni[i-1]

        sbeta = (s1 * omega[0] + np.sum([si[i-1] * omega[i] for i in range(1, d)]) + sb * b) / beta
        y.append([omega[0] + s1 * dt] + [omega[i] + si[i-1] * dt for i in range(1, d)] + [b + sb * dt, beta + sbeta * dt])
        t.append(t[-1] + dt)

        if t[-1] > T:
            break
        if np.isnan(np.sum(y[-1])):
            t.pop()
            y.pop()
            break

    return np.array(y).transpose(), np.array(t)

def execute(args):

    res = {
        "y0": list(),
        "t": list(),
        "w": list(),
        "b": list(),
        "beta": list()
        }

    for i in range(args.N):
        print(i)
        y0 = np.random.randn(4)
        sol = model(y0, args.dt, args.d, args.a, args.T, args.xmin, args.xmax, args.p, args.Ni)
        res["y0"].append(y0)
        res["t"].append(sol[1])
        res["w"].append(sol[0][:d])
        res["b"].append(sol[0][d])
        res["beta"].append(sol[0][d+1])

    return res

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--T", type=float, required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--p", type=int, default=-1)
    parser.add_argument("--xmin", type=float, default=-0.3)
    parser.add_argument("--xmax", type=float, default=1.18549)
    parser.add_argument("--a", type=float, default=1.)
    parser.add_argument("--Ni", nargs='+', type=float)
    parser.add_argument("--pickle", type=str, required=True)
    args = parser.parse_args()

    torch.save(args, args.pickle)
    saved = False
    try:
        res = execute(args)
        with open(args.pickle, 'wb') as f:
            torch.save(args, f)
            torch.save(res, f)
            saved = True
    except:
        if not saved:
            os.remove(args.pickle)
        raise


if __name__ == "__main__":
    main()
