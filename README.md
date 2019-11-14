# Feature and lazy training

[article on arXiv](https://arxiv.org/abs/1906.08034)

## notation
The notations in the code does not match with the notation in the article,
- the scaling factor at the network's output is `alpha/h` in the code and `alpha/sqrt(h)` in the article
- the loss prefactor is `1/alpha` in the code and `1/alpha^2` in the article

it imply that `alpha_code = sqrt(h) alpha_article` and `t_code/h = t_article/(sqrt(h) alpha_article)`.

## dependancies
- pytorch
- github.com/mariogeiger/hessian
- github.com/mariogeiger/grid

## example
```
python -m grid F10k3Lsp_alpha --n 1 "python main.py --train_time 18000 --data_seed 0 --batch_seed 0 --init_kernel 0 --final_kernel 0 --delta_kernel 0 --arch fc_softplus --L 3 --dataset fashion --ptr 10000 --pte 50000 --tau_alpha_crit 1e3 --tau_over_h 1e-3" --init_seed 0 1 2 3 4 5 6 7 8 9 --alpha 1e-4 1e-2 1e0 1e2 1e4 1e6 --h:int 100 1000
```

how to read the output file:
```python
from grid import load
runs = load('F10k3Lsp_alpha')

for r in runs:
    print(r['regular']['test']['outputs'])  # display output of the testset
    print(r['regular']['dynamics'][-1]['test']['err'])  # display the final test error
```
