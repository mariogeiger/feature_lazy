# feature and lazy learning

dependancies:
- pytorch
- github.com/mariogeiger/hessian
- github.com/mariogeiger/grid

examples:
```
python -m grid F10k3L --n 1 "python main.py --device cpu --train_time 18000 --seed_trainset 0 --seed_testset 0 --arch fc --act swish --L 3 --dataset fashion --ptr 10000 --pte 50000 --tau_alpha_crit 1e3 --tau_over_h 1e-3" --seed_init 0 1 2 3 4 5 6 7 8 9 --alpha 1e-4 1e-2 1e0 1e2 1e4 1e6 --h:int 100 150
```

how to read the output file:
```python
from grid import load
runs = load('F10k3L')

print("F(testset) for the first run loaded")
print(runs[0]['regular']['test']['outputs'])

alphas = sorted({r['args'].alpha for r in runs})

for alpha in alphas:
    rs = [r for r in runs if r['args'].alpha == alpha]
    print("{} runs with alpha = {}".format(len(rs), alpha))

    print("mean error = {}".format(sum(r['regular']['dynamics'][-1]['test']['err']) / len(rs)))
```
