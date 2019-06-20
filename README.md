# feature and lazy learning

dependancies:
- pytorch
- github.com/mariogeiger/hessian
- github.com/mariogeiger/grid (optional)

examples:
```
python main.py --arch fc_relu --data_seed 0 --batch_seed 0 --d 10 --L 5 --dataset mnist --ptr 5000 --max_bs 5000 --pte 50000 --train_time 3600 --temp 0 --tau -1000 --init_kernel 0 --final_kernel 0 --alpha 0.001 --init_seed 0 --h 90 --pickle output.pkl

python main.py --arch cv_relu --data_seed 0 --batch_seed 0 --dataset mnist --ptr 500 --max_bs 32 --pte 500 --train_time 3600 --temp 1e-6 --tau -10 --init_kernel 0 --final_kernel 0 --alpha 0.001 --init_seed 0 --h 10 --pickle output.pkl --device cpu
```

how to read the output file:
```python
path = 'output.pkl'

with open(path, 'rb') as f:
    args = torch.load(f)
    try:
        results = torch.load(f, map_location='cpu')
    except EOFError:
        print('no result in that file')

print(args)
print(results['regular']['test']['outputs'])
```
