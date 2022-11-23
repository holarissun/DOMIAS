# Code for DOMIAS

running: 
1. to reproduce our restuls and evaluate eqn.1 and eqn.2 on specified dataset, run
```python
python3 domias_main.py --seed 0 --gpu_idx 0 --flows 5 --gan_method TVAE --dataset housing --training_size_list 30 50 100 300 500 1000 --held_out_size_list 10000 --gen_size_list 10000 --training_epoch_list 2000
```

or equivalently, run 
```python
bash run_tabular.sh
```

2. if using prior knowledge (i.e., no reference dataset setting), add
```python
--density_estimator prior
```

3. To run experiment with the CelebA dataset, first run 
```python
python3 celeba_gen.py --gpu_idx 0 --seed 0 --training_size 4000
```
and then
```python
python3 celeba_eval.py --gpu_idx 0 --seed 0 --training_size 4000
```
