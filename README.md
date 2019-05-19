# mf_ntk_article

dependancies:
- pytorch
- github.com/mariogeiger/hessian
- github.com/mariogeiger/grid (optional)

example:
```
python main.py --arch fc_relu --data_seed 0 --batch_seed 0 --d 10 --L 5 --dataset mnist --ptr 5000 --chunk 5000 --pte 50000 --train_time 36000 --desc "" --temp 0 --tau -1000 --init_kernel 0 --final_kernel 0 --alpha 0.001 --init_seed 0 --h 90 --pickle output.pkl
```