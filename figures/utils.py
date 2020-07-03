import os
import sys
import glob
import torch
import numpy as np
from scipy.special import erfc
from scipy.integrate import quad

#sys.path.append("/home/jonas/Documents/nn/feature_lazy/")
sys.path.append("/home/jpaccola/feature_lazy/")
from main import SplitEval
from arch import FC, FixedAngles
from kernels import compute_kernels

#path = "/home/jonas/Documents/nn/result/clean/NN/"
path = "/home/jpaccola/compressing_invariant_manifolds_in_neural_nets/result/"

def test(a, dic):

    if dic is None:
        return True
    for key in dic.keys():
        if not getattr(a, key) == dic[key]:
            return False

    return True

def load_data(name, dic, mul=False, action=None):

    out = list()
    for filename in glob.glob("{}/{}/*.pkl".format(path, name)):
        with open(filename, 'rb') as f:
            try:
                a = torch.load(f, map_location="cpu")
                if test(a, dic):
                    r = torch.load(f, map_location="cpu")
                    out_ = (a, r)
                    if action is not None:
                        out_ = action(out_)
                    if mul:
                        out.append(out_)
                    else:
                        return out_
            except EOFError:
                pass
    return out

def get_argument(name, arg, dic=None):

    xs = list()
    for filename in glob.glob("{}/{}/*.pkl".format(path, name)):
        with open(filename, 'rb') as f:
            a = torch.load(f, map_location="cpu")
            if test(a, dic):
                xs.append(getattr(a, arg))
    xs = list(set(xs))
    try:
        xs.sort()
    except:
        pass

    return xs

def get_mean(data):

    ps = np.array([p for p in data.keys()])
    means = list()

    for p in ps:
        means.append(np.mean(data[p]))
    means = np.array(means)

    return ps, means

def get_median(data):

    ps = np.array([p for p in data.keys()])
    medians = list()

    for p in ps:
        medians.append(np.median(data[p]))
    medians = np.array(medians)

    return ps, medians

def get_logmean(data):

    ps = np.array([p for p in data.keys()])
    logmeans = list()

    for p in ps:
        logmeans.append(np.mean(np.log10(data[p])))
    logmeans = np.array(logmeans)

    return ps, 10**logmeans

def get_std(data):

    ps = np.array([p for p in data.keys()])
    stds = list()

    for p in ps:
        stds.append(np.std(data[p]))
    stds = np.array(stds)

    return ps, stds

def NNfunc(args, network):

    act = lambda x: 2 ** 0.5 * torch.relu(x)
    if args.arch == "fc":
        f = FC(args.d, args.h, 1, args.L, act, args.bias, args.last_bias, args.var_bias).to("cpu")
    elif args.arch == "fixed_angles":
        f = FixedAngles(args.d, args.h, act, args.bias)
    else:
        print("invalid architecture")
    f = SplitEval(f, args.chunk)
    f.load_state_dict(network)
    f.eval()

    return f

def Kcircle_map(f, N=100, R=2**0.5):

    phi = torch.linspace(-np.pi, np.pi, N)
    x = torch.zeros(N, 2)
    x[:, 0] = phi.cos() * R
    x[:, 1] = phi.sin() * R
    k, _, _ = compute_kernels(f, x, x[:1])

    return phi.numpy(), k.numpy()

def cleanup(sol):
    new_sol = list()
    for _sol in sol:
        if np.std(_sol[0][0,-2:]) > 1e2:
            new_sol.append(_sol)
    return new_sol

def average(sol, j):

    N = len(sol)
    n = max([len(sol[i][0][0]) for i in range(N)])
    w = np.zeros(n)
    dw = np.zeros(n)
    r = np.zeros(n)

    for i in range(N):
        ni = len(sol[i][0][j])
        w[:ni] += sol[i][0][j]**2
        dw[:ni] += (sol[i][0][j] - sol[i][0][j, 0])**2
        r[:ni] += np.ones(ni)

    return (w / r)**0.5, (dw / r)**0.5

def S1func0(p, d, xmin=-0.3, xmax=1.18549):

    pf = np.exp(-xmin**2 / 2) - np.exp(-xmax**2 / 2)
    mu12 = pf**2 * 2 / (3 * np.pi) * d / d
    sg12 = 0.5

    return mu12

def Sjfunc1(p, d, xmin=-0.3, xmax=1.18549):

    pf = np.exp(-xmin**2 / 2) - np.exp(-xmax**2 / 2)
    muj2 = pf**2 / (3**1.5 * np.pi**2 * d**2)
    sgj2 = 0.5

    return muj2 + (sgj2) / p

def lambda_ODEend(p, xmin=-0.3, xmax=1.18549):
    return (2 / np.pi * (np.exp(-xmin**2 / 2) - np.exp(-xmax**2 / 2))**2 * p)**0.5

def model(y0, dt, d, p, h=10000, xmin=-0.3, xmax=1.18549, stop=1, T=None, ap=True):

    a = (2 / d) ** 0.5 / h
    C0 = np.exp(-xmin**2 / 2) - np.exp(-xmax**2 / 2)
    c0 = y0[0]**2 + (d-1) * y0[1]**2 + y0[2]**2 - y0[3]**2
    c = c0

    y = [y0]
    t = [0]
    while True:
        omega1, omegaj, b, beta = y[-1]
        c = omega1**2 + (d-1) * omegaj**2 + b**2 - beta**2
        mu1 = 1 / (2 * np.pi)**0.5 * erfc(-b/2**0.5) * C0
        muj = - b * omega1 * omegaj / np.pi / d * np.exp(-b**2 / 2) * C0
        mub = omega1 / np.pi * np.exp(-b**2 / 2) * C0
        sigma2 = 0.5 * erfc(-b / 2**0.5)

        s1 = a * beta * mu1
        sj = a * beta * (muj + (sigma2 / p)**0.5)
        sb = a * beta * mub
        if ap:
            sb += a * beta * (sigma2 * d / p)**0.5

        beta = (omega1**2 + (d-1) * omegaj**2 + b**2 - c0)**0.5 * np.sign(y0[3])
        y.append([omega1 + s1 * dt, omegaj + sj * dt, b + sb * dt, beta])
        t.append(t[-1] + dt)

        if stop is not None and np.abs((c - c0) / c0) > stop:
            break
        if T is not None and t[-1] > T:
            break
        if np.isnan(np.sum(y[-1])):
            t.pop()
            y.pop()
            break

    return np.array(y).transpose(), np.array(t)

def get_tstar(args, dy, frac=0.1):
    t = np.array([x["t"] for x in dy])
    margin_frac = np.array([(args.ptr - x["train"]["nd"]) / args.ptr for x in dy])
    for t_, m in zip(t, margin_frac):
        if m > frac:
            return t_

def get_lc(name, dic, keys, err_max=0.25):

    import copy
    dic_ = dic.copy()
    ps = get_argument(name, "ptr", dic_)

    data = {
        "regular": dict(),
        "init": dict(),
        "final": dict(),
        "stretch": dict(),
    }

    for p in ps:
        dic_["ptr"] = p
        for key, x in data.items(): x[p] = list()
        for seed in get_argument(name, "seed_trainset", dic_):
            dic_["seed_trainset"] = seed
            try:
                a, r = load_data(name, dic_)
                if r["finished"]:
                    err = r["regular"]["dynamics"][-1]["test"]["err"]
                    if err < err_max:
                        data["regular"][p].append(err)
                    if "init" in keys:
                        err = r["init_kernel_ptr"]["dynamics"][-1]["test"]["err"]
                        if err < err_max:
                            data["init"][p].append(err)
                    if "final" in keys:
                        err = r["final_kernel_ptr"]["dynamics"][-1]["test"]["err"]
                        if err < err_max:
                            data["final"][p].append(err)
                    # if "stretch" in keys:
                    #     err1 = r["stretch_kernel"]["NN"]["dynamics"][-1]["test"]["err"]
                    #     err2 = r["stretch_kernel"]["ODE"]["dynamics"][-1]["test"]["err"]
                    #     if err1 < err_max:
                    #         data["stretch_NN"][p].append(err1)
                    #     if err2 < err_max:
                    #         data["stretch_ODE"][p].append(err2)
                    if "stretch" in keys:
                        err = r["stretch_kernel"]["dynamics"][-1]["test"]["err"]
                        if err < err_max:
                            data["stretch"][p].append(err)
            except:
                pass
        dic_ = dic.copy()

    return data

def get_amplification_factor_at_tstar(name, dic, frac=0.1):

    def action(out):
        a, r = out
        for x in r["regular"]["dynamics"]:
            if (a.ptr - x["train"]["nd"]) / a.ptr > frac:
                return x["w"][0] / torch.tensor(x["w"][1:]).float().mean()

    data = dict()
    for p in get_argument(name, "ptr", dic):
        dic["ptr"] = p
        data[p] = load_data(name, dic, mul=True, action=action)

    return data

def f_C1(z, label, xmin, xmax, d, pos):
    m = lambda x: label(x, xmin, xmax) * np.exp(-x**2/2) / (2*np.pi)**0.5 * x
    if pos:
        return quad(m, z, np.infty)[0]
    else:
        return quad(m, -np.infty, z)[0]

def f_Cb(z, label, xmin, xmax, d, pos):
    m = lambda x: label(x, xmin, xmax) * np.exp(-x**2/2) / (2*np.pi)**0.5
    if pos:
        return quad(m, z, np.infty)[0]
    else:
        return quad(m, -np.infty, z)[0]
