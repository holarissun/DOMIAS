import argparse

parser = argparse.ArgumentParser()

# Optim params
parser.add_argument('--out_dim', type=int, default=10, help = 'dimension of each output')
parser.add_argument('--in_dim', type=int, default=20, help = 'dimension of input')
parser.add_argument('--data_size', type=int, default=30000, help = 'size of generated dataset')

args = parser.parse_args()

from metrics.combined import compute_metrics
# build three random networks as random functions (mappings)
import torch
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

class RandNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RandNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, hidden_state):
        return self.layers(hidden_state)

col1, col2, col3 = args.out_dim, args.out_dim, args.out_dim
f1 = RandNet(args.in_dim, col1)
f2 = RandNet(args.in_dim, col2)
f3 = RandNet(args.in_dim, col3)
f_back_pred = RandNet(col1+col2+col3, args.in_dim) # backward prediction

# generate dataset

D_SIZE = args.data_size
dataset = np.zeros((D_SIZE, col1 + col2 + col3))
label = np.zeros((D_SIZE, args.in_dim)) # the gaussian noise input

testset = np.zeros((D_SIZE, col1 + col2 + col3))
test_label = np.zeros((D_SIZE, args.in_dim)) # the gaussian noise input


for i in range(D_SIZE):
    rand_hidden_state = torch.normal(mean=torch.arange(-1., 1., 0.1), std=torch.arange(1, -1, -0.1))
    label[i] = rand_hidden_state
    dataset[i] = torch.cat((f1(rand_hidden_state).detach(), f2(rand_hidden_state).detach(), f3(rand_hidden_state).detach()), dim=-1).numpy()
    
for i in range(D_SIZE):
    rand_hidden_state = torch.normal(mean=torch.arange(-1., 1., 0.1), std=torch.arange(1, -1, -0.1))
    test_label[i] = rand_hidden_state
    testset[i] = torch.cat((f1(rand_hidden_state).detach(), f2(rand_hidden_state).detach(), f3(rand_hidden_state).detach()), dim=-1).numpy()

### Save dataset
import os
os.makedirs('dataset', exist_ok=True)
np.save(f'dataset/synthetic_gaussian_{args.in_dim}_{args.out_dim}_{args.data_size}_train.npy', dataset)
np.save(f'dataset/synthetic_gaussian_{args.in_dim}_{args.out_dim}_{args.data_size}_test.npy', testset)

### Evaluation:
start_time = time.time()
compute_metrics(dataset, testset)
print('time for evaluation:', time.time() - start_time)


