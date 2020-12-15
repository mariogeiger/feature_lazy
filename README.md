# feature and lazy learning

## dependancies
- pytorch
- github.com/mariogeiger/hessian
- github.com/mariogeiger/grid

## commands for MNIST
```
# folder containing 3 scans in h for 3 values of alpha (stop runs after 2 hours)
python -m grid pcaM10d5kp2Lsw10_h "python main.py --dataset pca_mnist --d 10 --pte 40000 --arch fc --max_wall 7200 --L 2 --act swish --act_beta 10 --ckpt_step 500 --delta_kernel 1 --ptk 2500 --ptr 5000" --seed_init 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 --alpha 0.00390625 12 1048576 --h:int 6 7 8 9 10 11 12 14 16 18 20 22 24 28 32 36 40 44 48 56 64 72 80 88 96 112 128 144 160 176 192 224 256 288 320 352 384 448 512 576 640 704 768 896 1024 1152 1280 1408

# folder used for computing the ensemble average generalization error
python -m grid pcaM10d5kp2Lsw10_h_alpha_grid "python main.py --dataset pca_mnist --d 10 --pte 40000 --arch fc --max_wall 1200 --L 2 --act swish --act_beta 10 --ckpt_step 500 --delta_kernel 1 --ptk 2500 --ptr 5000" --seed_init 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 --alpha 0.00031280517578125 0.001007080078125 0.003173828125 0.010009765625 0.03125 0.1015625 0.3125 1.0 3.125 10.0 32 100 320 992 3200 9984 31744 98304 319488 1015808 3145728 9961472 31457280 --h:int 20 25 31 37 45 54 66 82 100 120 148 176 216 264 320 384 480 576 704 864 1024 1248

# folder used to localize the jamming transition and the feature/lazy crossover
python -m grid pcaM10d5kp2Lsw10_h_alpha "python main.py --dataset pca_mnist --d 10 --pte 40000 --arch fc --max_wall 900 --L 2 --act swish --act_beta 10 --ckpt_step 500 --delta_kernel 1 --ptk 2500" --ptr 5000 --seed_init 0 1 2 --alpha 0.00031280517578125 0.001007080078125 0.003173828125 0.010009765625 0.03125 0.1015625 0.3125 1.0 3.125 10.0 32 100 320 992 3200 9984 31744 98304 319488 1015808 3145728 9961472 31457280 --h:int 20 25 31 37 45 54 66 82 100 120 148 176 216 264 320 384 480 576 704 864 1024 1248
# scripts jamming.py and feature_lazy.py were also used to generate this data folder
```

## commands for CIFAR10
```
# scan in h
python -m grid pcaC30d5kp2Lsw10_h "python main.py --dataset pca_cifar10 --d 30 --pte 40000 --arch fc --max_wall 7200 --L 2 --act swish --act_beta 10 --ckpt_step 500 --delta_kernel 1 --ptk 2500 --ptr 5000" --seed_init 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 --alpha 2 --h:int 9 10 11 12 14 16 18 20 22 24 28 32 36 40 44 48 56 64 72 80 88 96 112 128 144 160 176 192 224 256 288 320 352 384 448 512 576 640 704 768 896 1024 1152 1280

# generalization error
python -m grid pcaC30d5kp2Lsw10_h_alpha_grid "python main.py --dataset pca_cifar10 --d 30 --pte 40000 --arch fc --max_wall 1500 --L 2 --act swish --act_beta 10 --ckpt_step 500 --delta_kernel 1 --ptk 2500 --ptr 5000" --seed_init 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 --alpha 0.000244140625 0.00048828125 0.0009765625 0.001953125 0.00390625 0.0078125 0.015625 0.03125 0.0625 0.125 0.25 0.5 1.0 2.0 4.0 8.0 16.0 32.0 64.0 128.0 256.0 512.0 1024 2048 4096 8192 16384 32768 --h:int 32 40 48 64 80

# transitions
python -m grid pcaC30d5kp2Lsw10_h_alpha "python main.py --dataset pca_cifar10 --d 30 --pte 40000 --arch fc --max_wall 900 --L 2 --act swish --act_beta 10 --ckpt_step 500 --delta_kernel 1 --ptk 2500 --ptr 5000" --seed_init 0 --alpha 0.0001220703125 0.000244140625 0.00048828125 0.0009765625 0.001953125 0.00390625 0.0078125 0.015625 0.03125 0.0625 0.125 0.25 0.5 1.0 2.0 4.0 8.0 16.0 32.0 64.0 128.0 256.0 512.0 1024.0 2048.0 4096.0 8192.0 16384.0 --h:int 20 24 32 40 48 64 80 96 128 160 192 256 320 384 512 640 768 1024 1280 1536
# scripts jamming.py and feature_lazy.py were also used to generate this data folder
```

## code to plot the scan in h
```python
from grid import load, group_runs
import matplotlib.pyplot as plt
import math
import functools
import copy
from collections import Counter

@functools.lru_cache()
def loaddata(path):
    runs = load(path, cache=False)
    runs = [
        {
            'regular': {
                'dynamics': [
                    {
                        'wall': x['wall'],
                        'train': {
                            'nd': x['train']['nd']
                        }
                    }
                    for x in r['regular']['dynamics'][:-1]
                ] + r['regular']['dynamics'][-1:]
            },
            'args': r['args'],
            'finished': r['finished'],
            'delta_kernel': r['delta_kernel'] if 'delta_kernel' in r else None,
        }
        for r in runs
    ]
    return copy.deepcopy(runs)
    
def mean(xs):
    return sum(xs) / len(xs)

def last(it):
    return [x for x in it][-1]

def texnum(x, mfmt='{}', noone=False):
    m, e = "{:e}".format(x).split('e')
    m, e = float(m), int(e)
    mx = mfmt.format(m)
    if e == 0:
        if m == 1:
            return "" if noone else "1"
        return mx
    ex = "10^{{{}}}".format(e)
    if m == 1:
        return ex
    return "{}\;{}".format(mx, ex)

def enserr(xs):
    o = mean([x['outputs'] for x in xs])
    y = mean([x['labels'] for x in xs])
    return (o * y).lt(0).double().mean().item()

# plt.gca().yaxis.set_major_formatter(format_percent)
import matplotlib.ticker as ticker
@ticker.FuncFormatter
def format_percent(x, _pos=None):
    x = 100 * x
    if abs(x - round(x)) > 0.05:
        return r"${:.1f}\%$".format(x)
    else:
        return r"${:.0f}\%$".format(x)
    
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(6.4, 3), dpi=150)

for ax, path in zip([ax1, ax2], ['pcaM10d5kp2Lsw10_h', 'pcaC30d5kp2Lsw10_h']):
    plt.sca(ax)
    runs = loaddata(path)
    args, groups = group_runs(runs, group_by=['seed_init', 'h'])

    for ar, rs in groups:
        if 'alpha' in args:
            ar['alpha'] = args['alpha']
        rs = [r for r in rs if r['regular']['dynamics'][-1]['train']['err'] < 0.4]
        rs = [r for r in rs if math.isfinite(r['regular']['dynamics'][-1]['train']['aloss'])]
        if len(rs) <= 5: continue

        label = r"$\tilde \alpha = {}$".format(texnum(ar['alpha'], '{:.0f}'))

        [line] = plt.plot(
            [r['args'].h for r in rs], 
            [r['regular']['dynamics'][-1]['test']['err'] for r in rs], 
            '.', alpha=0.1)

        hs = sorted({r['args'].h for r in rs})
        y = [mean([r['regular']['dynamics'][-1]['test']['err'] for r in rs if r['args'].h == h]) for h in hs]
        plt.plot(hs, y, '--', color=line.get_color())

        rs = [r for r in rs if r['regular']['dynamics'][-1]['test']['outputs'] is not None]
        hs = Counter([r['args'].h for r in rs])
        n = [c for h, c in hs.most_common()][math.floor(0.9 * len(hs))]
        hs = sorted([h for h, c in hs.items() if c >= n])
        print("ensemble over {}".format(n))

        h1 = next(h for h in hs 
                  if len([r for r in rs
                          if r['args'].h == h and r['regular']['dynamics'][-1]['train']['nd'] == 0]) >= 2)
        h2 = last(h for h in hs 
                  if len([r for r in rs 
                          if r['args'].h == h and r['regular']['dynamics'][-1]['train']['nd'] > 0]) >= 2)

        plt.plot([h1, h1], [0, 1], '--', color=line.get_color(), linewidth=0.5)
        plt.plot([h2, h2], [0, 1], '--', color=line.get_color(), linewidth=0.5)
        plt.fill_betweenx([0, 1], [h1, h1], [h2, h2], color=line.get_color(), alpha=0.1, zorder=-10)

        y = [enserr([r['regular']['dynamics'][-1]['test'] for r in rs if r['args'].h == h][:n]) for h in hs]
        plt.plot(hs, y, '-', color=line.get_color(), label=label)

    plt.xscale('log')
    plt.legend()

    plt.xlabel(r'$h$')
    plt.xlim(0.95 * min(hs), max(hs))
    plt.gca().yaxis.set_major_formatter(format_percent)

plt.sca(ax1)
plt.ylabel(r'$\epsilon$')
plt.ylim(0.055, 0.16)
plt.title('MNIST 10 PCA')

plt.sca(ax2)
plt.ylim(0.25, 0.36)
plt.title('CIFAR10 30 PCA')

plt.tight_layout()
plt.savefig('overfitting.pgf')
```
![image](https://user-images.githubusercontent.com/333780/102210800-f33f5380-3ed2-11eb-8fb0-605ebd3888df.png)

## code to plot the phase diagram
```python
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(6.4, 3), dpi=150)

plt.sca(ax1)
for a in [2**-8, 12, 2**20]:
    plt.plot([6, 1e4], [a]*2, '-.', linewidth=0.5)

plt.sca(ax2)
for a in [2]:
    plt.plot([6, 1e4], [a]*2, '-.', linewidth=0.5)
plt.plot([], [])
plt.plot([], [])

for ax, path in zip([ax1, ax2], ['pcaM10d5kp2Lsw10_h_alpha', 'pcaC30d5kp2Lsw10_h_alpha']):
    plt.sca(ax)

    runs = loaddata(path + '_grid')
    runs = [r for r in runs if r['regular']['dynamics'][-1]['train']['nd'] <= 0.05 * r['args'].ptr]
    runs = [r for r in runs if r['finished']]

    cou = Counter([(r['args'].alpha, r['args'].h) for r in runs])
    nens = 15
    nens = Counter([c for _, c in cou.items()]).most_common()[0][0]
    print('ensemble on {} runs'.format(nens))
    ahs = sorted([ah for ah, c in cou.items() if c >= nens])

    errs = [
        enserr([
            r['regular']['dynamics'][-1]['test'] 
            for r in runs if r['args'].alpha == a and r['args'].h == h
        ][:nens])
        for a, h in ahs
    ]

    vor = Voronoi(
        [[5 * math.log2(h), math.log2(a)] for a, h in ahs] + [
            [0, 100], [0, -100], [200, 100], [200, -100]
        ]
    )
    cmap = mpl.cm.YlOrBr
    rmin, rmax = min(errs), max(errs)
    for err, i in zip(errs, vor.point_region):
        region = vor.regions[i]
        if not -1 in region:
            polygon = [[2**(vor.vertices[i][0] / 5), 2**vor.vertices[i][1]] for i in region]
            plt.fill(
                *zip(*polygon), 
                color=cmap((err - rmin) / (rmax - rmin)),
                zorder=-10,
            )
    print(r"from {:.3} to {:.3}".format(rmin, rmax))


    # FEATURE LAZY
    runs = loaddata(path)
    runs = [r for r in runs if r['regular']['dynamics'][-1]['train']['nd'] <= 0]
    runs = [r for r in runs if r['finished']]

    nfl = 4
    for c in [1e-4, 1e-2, 1, 100]:
        feature_lazy = []

        hs = sorted({r['args'].h for r in runs})
        for h in hs:
            rr = [r for r in runs if r['args'].h == h]
            als = sorted([x for x, c in Counter([r['args'].alpha for r in rr]).items() if c >= nfl])
            if len(als) < 2: continue

            def islazy(a):
                rs = [r for r in rr if r['args'].alpha == a]
                xs = [r['delta_kernel']['traink'] / r['delta_kernel']['init']['traink']['norm'] for r in rs]
                return 2**mean([math.log2(x) for x in xs]) < c

            if islazy(als[-1]) and not islazy(als[0]):
                a1, a2 = last(a for a in als if not islazy(a)), next(a for a in als if islazy(a))
                if a2 / a1 < 3:
                    feature_lazy.append((h, a1, a2))

        if not feature_lazy:
            continue

        plt.plot(
            [h for h, a1, a2 in feature_lazy], 
            [(a1*a2)**0.5 for h, a1, a2 in feature_lazy], 
            '-' if c == 1 else '--',
            color='k'
        )
        h = 300
        _, a1, a2 = sorted(feature_lazy, key=lambda x: abs(x[0] - h))[0]
        a = (a1 * a2)**0.5
        plt.annotate(
            r'$\| \Delta \Theta \| = {} \| \Theta_0 \|$'.format(texnum(c, noone=True)), 
            xy=(h, a), 
            horizontalalignment='center', 
            verticalalignment='bottom',
            fontsize=7,
        )


    # JAMMING
    runs = loaddata(path)
    runs = [r for r in runs if r['finished']]

    njam = 3
    for wall in [7200/4, 7200/2, 7200]:
        jamming = []

        als = sorted({r['args'].alpha for r in runs})
        for a in als:
            rr = [r for r in runs if r['args'].alpha == a]

            def finished(r, wall):
                x = r['regular']['dynamics'][-1]
                if x['wall'] >= wall:
                    x = next(x for x in r['regular']['dynamics'] if x['wall'] >= wall)
                    return x
                if x['train']['nd'] == 0:
                    return x
                return None

            rr = [r for r in rr if finished(r, wall) is not None]
            hs = sorted({r['args'].h for r in rr})

            def isjammed(h):
                xs = [finished(r, wall) for r in rr if r['args'].h == h]
                return all(x['train']['nd'] > 0 for x in xs) and len(xs) >= njam

            try:
                h1 = last(h for h in hs if isjammed(h))
                h2 = next(h for h in hs if h > h1 and not isjammed(h))
                if h2 - h1 <= 2:
                    jamming.append((h1, h2, a))
            except:
                pass

        if not jamming:
            continue

        plt.plot(
            [(h1*h2)**0.5 for h1, h2, a in jamming], 
            [a for h1, h2, a in jamming], 
            '-',
            label=r'{:.0f}min'.format(wall / 60),
            linewidth=1.2,
        )

    plt.fill_betweenx(
        [a for h1, h2, a in jamming], 
        [1] * len(jamming), 
        [(h1*h2)**0.5 for h1, h2, a in jamming],
        color='black'
    )

    if len(jamming) > 3:
        plt.ylim(min([a for h1, h2, a in jamming]), max([a for h1, h2, a in jamming]))

    runs = loaddata(path)
    args = args_intersection([r['args'] for r in runs])

    plt.yscale('log')
    plt.xscale('log')

    l = plt.legend(fontsize=8, frameon=False, handlelength=1, labelspacing=0.1, loc=0)
    for text in l.get_texts():
        text.set_color("white")

    print("{arch} L={L} {act}({act_beta:.0f}x) {dataset} d={d} p={ptr} gradflow {loss}({loss_beta:.0f}x)".format(**args))

    plt.annotate('lazy', xy=(3e2, 2e4), horizontalalignment='center', verticalalignment='center', fontsize=15)
    plt.annotate('feature', xy=(3e2, 2e-2), horizontalalignment='center', verticalalignment='center', fontsize=15)
    plt.annotate(
        'jamming', xy=(1.1e1, 1e1), 
        horizontalalignment='center', 
        verticalalignment='center', 
        fontsize=15, 
        rotation=90,
        color='white',
    )

    plt.xlabel(r'$h$', fontsize=13)

    plt.xlim(6, 1.5e3)

plt.sca(ax1)
plt.ylabel(r'$\tilde\alpha$', fontsize=13)
plt.title('MNIST 10 PCA')

plt.sca(ax2)
plt.title('CIFAR10 30 PCA')

plt.tight_layout()
plt.savefig('phasespace.pgf')
```

![image](https://user-images.githubusercontent.com/333780/102214054-f38e1d80-3ed7-11eb-8a74-45e0988c2f89.png)
