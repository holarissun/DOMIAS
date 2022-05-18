import argparse
import torch
parser = argparse.ArgumentParser()

# Optim params
parser.add_argument('--out_dim', type=int, default=10, help = 'dimension of each output')
parser.add_argument('--in_dim', type=int, default=20, help = 'dimension of input')
parser.add_argument('--data_size', type=int, default=30000, help = 'size of generated dataset')
parser.add_argument('--gan_method', type=str, default='TVAE', help = 'benchmarking generative model used for synthesis')
parser.add_argument('--training_size_list', nargs='+', type=int, default=[50, 500, 1000, 10000], help = 'size of training dataset')
parser.add_argument('--held_out_size_list', nargs='+', type=int, default=[50, 500, 1000, 10000], help = 'size of held-out dataset')
parser.add_argument('--training_epoch_list', nargs='+', type=int, default=[1, 10, 100, 1000, 2000], help = '# training epochs')
parser.add_argument('--gen_size_list', nargs='+', type=int, default=[100, 500, 1000, 5000, 10000], help = 'size of generated dataset')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument("--gpu_idx",default=None, type=int)

args = parser.parse_args()
# print(args.training_size_list)

if args.gpu_idx is not None:
    torch.cuda.set_device(args.gpu_idx)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import numpy as np
import pandas as pd
from ctgan import CTGANSynthesizer
from ctgan import load_demo
import ctgan
from sdv.tabular import TVAE
from sdv.evaluation import evaluate
from scipy import stats
from sklearn import metrics
from metrics.combined import compute_metrics
performance_logger = {}

''' 1. load dataset'''
dataset = np.load(f'dataset/synthetic_gaussian_{args.in_dim}_{args.out_dim}_{args.data_size}_train.npy') 

''' 2. training-test-addition split'''
for SIZE_PARAM in args.training_size_list:
    for ADDITION_SIZE in args.held_out_size_list:
        for TRAINING_EPOCH in args.training_epoch_list:
            performance_logger[f'{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}'] = {}
            training_set = dataset[:SIZE_PARAM]
            test_set = dataset[SIZE_PARAM:2*SIZE_PARAM]
            addtion_set = dataset[-ADDITION_SIZE:]

            ''' 3. Synthesis with TVAE'''
            df = pd.DataFrame(training_set)
            df.columns = [str(_) for _ in range(args.out_dim * 3)]
            if args.gan_method == 'TVAE':
                tvae_model = TVAE(epochs = TRAINING_EPOCH)
                tvae_model.fit(df)
            elif args.gan_method == 'CTGAN':
                tvae_model = CTGANSynthesizer(epochs=TRAINING_EPOCH)
                tvae_model.fit(df, split=False)
            for N_DATA_GEN in args.gen_size_list:
                samples = tvae_model.sample(N_DATA_GEN)
                #eval_met = evaluate(pd.DataFrame(samples), df)
                eval_met = compute_metrics(samples,dataset[:N_DATA_GEN], which_metric = ['WD'])['wd_measure']
                eval_ctgan = evaluate(pd.DataFrame(samples), df)
                performance_logger[f'{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}'][f'{N_DATA_GEN}_evaluation'] = (eval_met, eval_ctgan)
                print('SIZE: ', SIZE_PARAM, 
                      'TVAE EPOCH: ', TRAINING_EPOCH, 
                      'N_DATA_GEN: ', N_DATA_GEN, 
                      'ADDITION_SIZE: ', ADDITION_SIZE,
                      'Performance (Sample-Quality): ', eval_met, eval_ctgan)

                ''' 4. density estimation / evaluation of Eqn.(1) & Eqn.(2)'''
                density_gen = stats.gaussian_kde(samples.values.transpose(1,0))
                density_data = stats.gaussian_kde(addtion_set.transpose(1,0))

                # eqn1: \prop P_G(x_i)

                p_G_train = density_gen(training_set.transpose(1,0))
                p_G_test = density_gen(test_set.transpose(1,0))

                thres = np.quantile(np.vstack((p_G_train, p_G_test)), 0.5)
                
                auc_y = np.hstack((np.array([1]*training_set.shape[0]), np.array([0]*test_set.shape[0])))
                auc_pred = np.hstack((p_G_train, p_G_test))
                fpr, tpr, thresholds = metrics.roc_curve(auc_y, auc_pred, pos_label=1)
                auc = metrics.auc(fpr, tpr)
                
                print('Eqn.(1), training set prediction acc', (p_G_train > thres).sum(0) / SIZE_PARAM)
                print('Eqn.(1), test set prediction acc', (p_G_test > thres).sum(0) / SIZE_PARAM)
                print('Eqn.(1), AUC', auc)
                performance_logger[f'{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}'][f'{N_DATA_GEN}_Eqn1'] = (p_G_train > thres).sum(0) / SIZE_PARAM
                performance_logger[f'{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}'][f'{N_DATA_GEN}_Eqn1AUC'] = auc

                # eqn2: \prop P_G(x_i)/P_X(x_i)

                p_R_train = density_data(training_set.transpose(1,0))
                p_R_test = density_data(test_set.transpose(1,0))

                thres = np.quantile(np.vstack((p_G_train/p_R_train, p_G_test/p_R_test)), 0.5)

                auc_y = np.hstack((np.array([1]*training_set.shape[0]), np.array([0]*test_set.shape[0])))
                auc_pred = np.hstack((p_G_train/p_R_train, p_G_test/p_R_test))
                fpr, tpr, thresholds = metrics.roc_curve(auc_y, auc_pred, pos_label=1)
                auc = metrics.auc(fpr, tpr)
                
                print('Eqn.(2), training set prediction acc', (p_G_train/p_R_train > thres).sum(0) / SIZE_PARAM)
                print('Eqn.(2), test set prediction acc', (p_G_test/p_R_test > thres).sum(0) / SIZE_PARAM)
                print('Eqn.(2), AUC', auc)
                performance_logger[f'{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}'][f'{N_DATA_GEN}_Eqn2'] = (p_G_train/p_R_train > thres).sum(0) / SIZE_PARAM
                performance_logger[f'{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}'][f'{N_DATA_GEN}_Eqn2AUC'] = auc
                np.save(f'results_folder/performance_logger_{args.gan_method}_{args.seed}.npy',performance_logger)
