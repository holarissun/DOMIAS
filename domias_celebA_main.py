import argparse
import torch
parser = argparse.ArgumentParser()

parser.add_argument("--gpu_idx", default=0, type=int)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--rep_dim", type=int, default=128)
parser.add_argument("--ae_epoch", type=int, default=100)
parser.add_argument("--vae_epoch", type=int, default=1000)

args = parser.parse_args('')

GPU_IDX = args.gpu_idx
SEED = args.seed
LATENT_REPRESENTATION_DIM = args.rep_dim
AE_EPOCH = args.ae_epoch

alias = f'{SEED}_{GPU_IDX}_{LATENT_REPRESENTATION_DIM}'


import os 
import torch
import torch.nn as nn
import torchvision.transforms as tf 
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision.utils import save_image
from PIL import Image
%matplotlib inline
dirc='../subset_1000'
device = torch.device(f"cuda:{GPU_IDX}" if torch.cuda.is_available() else "cpu")

print(device)

workers=8
batch_size=128
image_size=64
ms=((0.5,0.5,0.5),(0.5,0.5,0.5))


dataset = ImageFolder(dirc,
                       transform=tf.Compose([
                           tf.Resize(image_size),
                           tf.CenterCrop(image_size),
                           tf.ToTensor(),
                           tf.Normalize(*ms),
                       ]))
# Create the dataloader
dataloader = DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)


def denormalized(img):
    return img * ms[1][0] + ms[0][0]


def deviceLoaderfunc(data,device):
    if isinstance(data,(list,tuple)):
        return [deviceLoaderfunc(x, device) for x in data]
    return data.to(device, non_blocking=True)
class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for b in self.dl: 
            yield deviceLoaderfunc(b, self.device)
            
    def __len__(self):
        return len(self.dl)
dataloader = DeviceDataLoader(dataloader, device)

import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class BetaVAE_H(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, z_dim=10, nc=3):
        super(BetaVAE_H, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256*1*1)),                 # B, 256
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            View((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 64, 64
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


class BetaVAE_B(BetaVAE_H):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    def __init__(self, z_dim=10, nc=1):
        super(BetaVAE_B, self).__init__()
        self.nc = nc
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 64, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),          # B,  32,  8,  8
            nn.ReLU(True),
            nn.Conv2d(256, 256, 4, 2, 1),          # B,  32,  4,  4
            nn.ReLU(True),
            View((-1, 256*4*4)),                  # B, 512
            nn.Linear(256*4*4, 256),              # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, z_dim*2),             # B, z_dim*2
            
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256*4*4),              # B, 512
            nn.ReLU(True),
            View((-1, 256, 4, 4)),                # B,  32,  4,  4
            nn.ConvTranspose2d(256, 256, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(64, nc, 4, 2, 1), # B,  nc, 64, 64
            nn.Tanh()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z).view(x.size())

        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

            
def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        #x_recon = F.tanh(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld



beta_vae = BetaVAE_B(z_dim=32, nc=3).to(device)

    
# def fit(epochs, lr, index=1):
lr = 0.002
epochs = args.vae_epoch
OBJ = 'H'
C_max = 25
C_stop_iter = 1e5

torch.cuda.empty_cache()
beta_vae_optimizer = torch.optim.Adam(beta_vae.parameters(), lr=lr, betas=(0.5, 0.999))

for epoch in range(epochs):
    for r_images, _ in tqdm(dataloader):
        x_recon, mu, logvar = beta_vae(r_images)
        recon_loss = reconstruction_loss(r_images, x_recon, 'gaussian')
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
        beta_vae_loss = recon_loss + 1.0*total_kld
    beta_vae_optimizer.zero_grad()
    beta_vae_loss.backward()
    beta_vae_optimizer.step()

    print("epoch:",epoch+1,"Loss:", beta_vae_loss.item(), 'recon loss:',recon_loss.item())

    
    

'''Auto-Encoder Training'''

encoder = nn.Sequential(
    # 3 x 64 x 64

    nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2, inplace=True),
    # 64 x 32 x 32

    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),
    # 128 x 16 x 16

    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),
    # 256 x 8 x 8

    nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),
    # 512 x 4 x 4

    nn.Conv2d(256, LATENT_REPRESENTATION_DIM, kernel_size=4, stride=1, padding=0, bias=False),
    # 1 x 1 x 1

    #nn.Flatten(),
    nn.Tanh()).to(device)

decoder = nn.Sequential(
    nn.ConvTranspose2d(LATENT_REPRESENTATION_DIM, 256, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
    # 512 x 4 x 4 upsampled data

    nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
    # 256 x 8 x 8

    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
    # 128 x 16 x 16

    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    # 64 x 32 x 32

    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
    # 3 x 64 x 64
).to(device)


def fit_AE(epochs, lr, index=1):
    
    torch.cuda.empty_cache()
    enc_dec_optimizer = torch.optim.Adam(
        [
            {"params": encoder.parameters(), "lr": lr, "betas":(0.5, 0.999)},
            {"params": decoder.parameters(), "lr": lr, "betas":(0.5, 0.999)},
        ]
        )
    #dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr, betas=(0.5, 0.999))
    
    for epoch in range(epochs):
        for r_images, _ in tqdm(dataloader):
#             loss_d, r_score, g_score = train_discriminator(r_images, d_optimizer)
#             loss_g = train_generator(g_optimizer)
            
            r_recover = decoder(encoder(r_images))
            loss = F.mse_loss(r_recover, r_images)
            loss.backward()
            enc_dec_optimizer.step()
        
        print("epoch:", epoch, " loss:", loss.item())
        
        
        
fit_AE(2, 0.0002)
HIDDEN = LATENT_REPRESENTATION_DIM
import numpy as np
N_SYNTH = 10000
synth_representation = np.zeros((N_SYNTH, HIDDEN))
if N_SYNTH > 50:
    for i in range(int(N_SYNTH/50)):
        #latent_noise = torch.randn(50, latent_size, 1, 1, device=device)
        latent_noise = torch.randn(50, beta_vae.z_dim, device=device)
        #beta_vae.decoder(latent_noise)#.shape
        g_images=beta_vae.decoder(latent_noise)
        synth_representation[i*50:(i+1)*50] = encoder(g_images).cpu().detach().squeeze().numpy()
        
        
np.save(f'celebA_representation/betavae_repres_synth_{alias}',synth_representation)





import numpy as np
N_REAL = 999
real_representation = np.zeros((N_REAL, HIDDEN))
i=0
for r_images, _ in tqdm(dataloader):
    real_representation[i*batch_size:i*batch_size+batch_size] = encoder(r_images).cpu().detach().squeeze().numpy()
    i+=1
    
np.save(f'celebA_representation/betavae_repres_real_{alias}', real_representation)
    
    
ref_dataset = ImageFolder('../subset_1000_10000',
                       transform=tf.Compose([
                           tf.Resize(image_size),
                           tf.CenterCrop(image_size),
                           tf.ToTensor(),
                           tf.Normalize(*ms),
                       ]))
# Create the dataloader
ref_dataloader = DataLoader(ref_dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
ref_dataloader = DeviceDataLoader(ref_dataloader, device)


import numpy as np
N_REF = 9000

ref_representation = np.zeros((N_REF, HIDDEN))
i=0
for r_images, _ in tqdm(ref_dataloader):
    ref_representation[i*batch_size:i*batch_size+batch_size] = encoder(r_images).cpu().detach().squeeze().numpy()
    i+=1
    
    
np.save(f'celebA_representation/betavae_repres_ref_{alias}', ref_representation)

test_dataset = ImageFolder('../subset_10000_11000',
                       transform=tf.Compose([
                           tf.Resize(image_size),
                           tf.CenterCrop(image_size),
                           tf.ToTensor(),
                           tf.Normalize(*ms),
                       ]))
# Create the dataloader
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
test_dataloader = DeviceDataLoader(test_dataloader, device)

import numpy as np
N_TEST = 1000

test_representation = np.zeros((N_TEST, HIDDEN))
i=0
for r_images, _ in tqdm(test_dataloader):
    test_representation[i*batch_size:i*batch_size+batch_size] = encoder(r_images).cpu().detach().squeeze().numpy()
    i+=1
    
np.save(f'celebA_representation/betavae_repres_test_{alias}', test_representation)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#alias = '0_2_128'

PATH_CELEB_REPRESENTATION = f'celebA_representation'
import argparse
import torch
parser = argparse.ArgumentParser()

parser.add_argument('--gan_method', type=str, default='adsgan-tf', choices = ["adsgan-tf","TVAE", "CTGAN", "KDE", 'gaussian_copula',
 'adsgan',
 'tvae',
 'privbayes',
 'marginal_distributions',
 'bayesian_network',
 'ctgan',
 'copulagan',
 'nflow',
 'rtvae',
 'pategan'], help = 'benchmarking generative model used for synthesis')
parser.add_argument('--epsilon_adsgan', type=float, default=0.1, help = 'hyper-parameter in ads-gan')
parser.add_argument('--density_estimator', type=str, default="bnaf", choices = ["bnaf", "kde"])
parser.add_argument('--training_size_list', nargs='+', type=int, default=[999], help = 'size of training dataset')
parser.add_argument('--held_out_size_list', nargs='+', type=int, default=[4500], help = 'size of held-out dataset')
parser.add_argument('--training_epoch_list', nargs='+', type=int, default=[2000], help = '# training epochs')
parser.add_argument('--gen_size_list', nargs='+', type=int, default=[5000], help = 'size of generated dataset')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument("--gpu_idx",default=0, type=int)
parser.add_argument("--device", type=str, default=None)
parser.add_argument(
    "--dataset",
    type=str,
    default="CelebA",
    choices=["MAGGIC","housing","synthetic","CelebA"],
)
parser.add_argument("--learning_rate", type=float, default=1e-2)
parser.add_argument("--batch_dim", type=int, default=50)
parser.add_argument("--clip_norm", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--patience", type=int, default=20)
parser.add_argument("--cooldown", type=int, default=10)
parser.add_argument("--early_stopping", type=int, default=100)
parser.add_argument("--decay", type=float, default=0.5)
parser.add_argument("--min_lr", type=float, default=5e-4)
parser.add_argument("--polyak", type=float, default=0.998)
parser.add_argument("--flows", type=int, default=5)
parser.add_argument("--layers", type=int, default=2) #
parser.add_argument("--hidden_dim", type=int, default=32)
parser.add_argument(
    "--residual", type=str, default="gated", choices=[None, "normal", "gated"]
)
parser.add_argument("--expname", type=str, default="")
parser.add_argument("--load", type=str, default=None)
parser.add_argument("--save", action="store_true")
parser.add_argument("--tensorboard", type=str, default="tensorboard")

args = parser.parse_args('')
args.device = f"cuda:{args.gpu_idx}"

if args.gpu_idx is not None:
    torch.cuda.set_device(args.gpu_idx)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.gan_method not in ["TVAE", "KDE", "CTGAN"]:
    from synthcity.plugins import Plugins
from bnaf.bnaf_den_est import *
from baselines_stable import *
from bnaf.datasets import *
import numpy as np
import pandas as pd
from ctgan import CTGANSynthesizer
from ctgan import load_demo
import ctgan
from sdv.tabular import TVAE
from sdv.evaluation import evaluate
from scipy import stats
from sklearn import metrics
from sklearn.metrics import accuracy_score
from metrics.combined import compute_metrics
import os
os.makedirs('results_folder', exist_ok = True)



#%% Import necessary functions
# from data_loader import load_adult_data
from adsgan import adsgan
from metrics.feature_distribution import feature_distribution
from metrics.compute_wd import compute_wd
from metrics.compute_identifiability import compute_identifiability
import numpy as np

#%% Experiment main function
def exp_main(args, orig_data_frame):
  
    orig_data = orig_data_frame
    # Generate synthetic data
    params = dict()
    params["lamda"] = args.lamda
    params["iterations"] = args.iterations
    params["h_dim"] = args.h_dim
    params["z_dim"] = args.z_dim
    params["mb_size"] = args.mb_size

    synth_data_list = adsgan(orig_data, params)
    synth_data = synth_data_list[0]
    print("Finish synthetic data generation")

    ## Performance measures
    # (1) Feature distributions
    feat_dist = feature_distribution(orig_data, synth_data)
    print("Finish computing feature distributions")

    # (2) Wasserstein Distance (WD)
    print("Start computing Wasserstein Distance")
    wd_measure = compute_wd(orig_data, synth_data, params)
    print("WD measure: " + str(wd_measure))

    # (3) Identifiability 
    identifiability = compute_identifiability(orig_data, synth_data)
    print("Identifiability measure: " + str(identifiability))

    return orig_data, synth_data_list, [feat_dist, wd_measure, identifiability]


performance_logger = {}

''' 1. load dataset'''
if args.dataset == "MAGGIC":
    dataset = MAGGIC('')
    dataset = np.vstack((dataset.train.x, dataset.val.x, dataset.test.x))
    np.random.shuffle(dataset)
    print('dataset size,', dataset.shape)
    ndata = dataset.shape[0]
elif args.dataset == "housing":
    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import StandardScaler
    def data_loader():
        #np.random.multivariate_normal([0],[[1]], n1)*std1 # non-training data
        scaler = StandardScaler() 
        X = fetch_california_housing().data
        np.random.shuffle(X)
        return scaler.fit_transform(X)
    dataset = data_loader()
    print('dataset size,', dataset.shape)
    ndata = dataset.shape[0]
elif args.dataset == "synthetic":
    dataset = np.load(f'{PATH_CELEB_REPRESENTATION}/dataset/synthetic_gaussian_{20}_{10}_{30000}_train.npy') 
    print('dataset size,', dataset.shape)
    ndata = dataset.shape[0]
elif args.dataset == "CelebA":
    training_set = np.load(f'{PATH_CELEB_REPRESENTATION}/betavae_repres_real_{alias}.npy')
    test_set = np.load(f'{PATH_CELEB_REPRESENTATION}/betavae_repres_test_{alias}.npy')[:999]
    addition_set = np.load(f'{PATH_CELEB_REPRESENTATION}/betavae_repres_ref_{alias}.npy')[:4500]
    addition_set2 = np.load(f'{PATH_CELEB_REPRESENTATION}/betavae_repres_ref_{alias}.npy')[4500:]
    
''' 2. training-test-addition split'''
for SIZE_PARAM in args.training_size_list:
    for ADDITION_SIZE in args.held_out_size_list:
        for TRAINING_EPOCH in args.training_epoch_list:
            performance_logger[f'{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}'] = {}
            if args.dataset != "CelebA":
                pass
            else:
                for N_DATA_GEN in args.gen_size_list:
                    samples = pd.DataFrame(np.load(f'{PATH_CELEB_REPRESENTATION}/betavae_repres_synth_{alias}.npy')[:N_DATA_GEN])
                    samples_val = pd.DataFrame(np.load(f'{PATH_CELEB_REPRESENTATION}/betavae_repres_synth_{alias}.npy')[N_DATA_GEN:N_DATA_GEN*2])
                #eval_met = evaluate(pd.DataFrame(samples), df)
                #eval_met = compute_metrics(samples, dataset[:N_DATA_GEN], which_metric = ['WD'])['wd_measure']
                wd_n = min(len(samples), len(addition_set))
                eval_met_on_held_out = compute_metrics(samples[:wd_n], addition_set[:wd_n], which_metric = ['WD'])['wd_measure']
                #eval_ctgan = evaluate(samples, pd.DataFrame(addition_set2))
                performance_logger[f'{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}'][f'{N_DATA_GEN}_evaluation'] = (eval_met_on_held_out)
                print('SIZE: ', SIZE_PARAM, 
                      'TVAE EPOCH: ', TRAINING_EPOCH, 
                      'N_DATA_GEN: ', N_DATA_GEN, 
                      'ADDITION_SIZE: ', ADDITION_SIZE,
                      'Performance (Sample-Quality): ',eval_met_on_held_out)

                ''' 4. density estimation / evaluation of Eqn.(1) & Eqn.(2)'''
                if args.density_estimator == "bnaf":
                    _gen, model_gen = density_estimator_trainer(samples.values, samples_val.values[:int(0.5*N_DATA_GEN)], samples_val.values[int(0.5*N_DATA_GEN):], args=args)
                    _data, model_data = density_estimator_trainer(addition_set, addition_set2[:int(0.5*ADDITION_SIZE)], addition_set2[:int(0.5*ADDITION_SIZE)], args=args)
                    p_G_train = compute_log_p_x(model_gen,torch.as_tensor(training_set).float().to(device)).cpu().detach().numpy()
                    p_G_test = compute_log_p_x(model_gen,torch.as_tensor(test_set).float().to(device)).cpu().detach().numpy()
                elif args.density_estimator == "kde":
                    density_gen = stats.gaussian_kde(samples.values.transpose(1,0))
                    density_data = stats.gaussian_kde(addition_set.transpose(1,0))
                    p_G_train = density_gen(training_set.transpose(1,0))
                    p_G_test = density_gen(test_set.transpose(1,0))

                X_test_4baseline = np.concatenate([training_set, test_set])
                Y_test_4baseline = np.concatenate([np.ones(training_set.shape[0]),np.zeros(test_set.shape[0])]).astype(bool)
                # build another GAN for hayes and GAN_leak_cal
                ctgan = CTGANSynthesizer(epochs=200)
                samples.columns = [str(_) for _ in range(training_set.shape[1])]
                ctgan.fit(samples) # train a CTGAN on the generated examples

                ctgan_representation = ctgan._transformer.transform(X_test_4baseline)
                print(ctgan_representation.shape)
                ctgan_score = ctgan._discriminator(torch.as_tensor(ctgan_representation).float().to(device)).cpu().detach().numpy()
                print(ctgan_score.shape)

                acc, auc = compute_metrics_baseline(ctgan_score, Y_test_4baseline)

                X_ref_GLC = ctgan.sample(addition_set.shape[0])

                baseline_results, baseline_scores = baselines(X_test_4baseline,
                                             Y_test_4baseline,
                                             samples.values,
                                             addition_set,
                                             X_ref_GLC
                                            )
                baseline_results = baseline_results.append({'name':'hayes', 'acc':acc, 'auc': auc}, ignore_index=True)
                baseline_scores['hayes'] = ctgan_score
                print('baselines:', baseline_results)
                performance_logger[f'{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}'][f'{N_DATA_GEN}_Baselines'] = baseline_results
                performance_logger[f'{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}'][f'{N_DATA_GEN}_BaselineScore'] = baseline_scores
                performance_logger[f'{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}'][f'{N_DATA_GEN}_Xtest'] = X_test_4baseline
                performance_logger[f'{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}'][f'{N_DATA_GEN}_Ytest'] = Y_test_4baseline

#                 eqn1: \prop P_G(x_i)
                log_p_test = np.concatenate([p_G_train, p_G_test])
                thres = np.quantile(log_p_test, 0.5)
                auc_y = np.hstack((np.array([1]*training_set.shape[0]), np.array([0]*test_set.shape[0])))
                fpr, tpr, thresholds = metrics.roc_curve(auc_y, log_p_test, pos_label=1)
                auc = metrics.auc(fpr, tpr)

                print('Eqn.(1), training set prediction acc', (p_G_train > thres).sum(0) / SIZE_PARAM)
                print('Eqn.(1), AUC', auc)
                performance_logger[f'{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}'][f'{N_DATA_GEN}_Eqn1'] = (p_G_train > thres).sum(0) / SIZE_PARAM
                performance_logger[f'{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}'][f'{N_DATA_GEN}_Eqn1AUC'] = auc
                performance_logger[f'{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}'][f'{N_DATA_GEN}_Eqn1Score'] = log_p_test
                # eqn2: \prop P_G(x_i)/P_X(x_i)
                if args.density_estimator == "bnaf":
                    p_R_train = compute_log_p_x(model_data,torch.as_tensor(training_set).float().to(device)).cpu().detach().numpy()
                    p_R_test = compute_log_p_x(model_data,torch.as_tensor(test_set).float().to(device)).cpu().detach().numpy()
                elif args.density_estimator == "kde":
                    p_R_train = density_data(training_set.transpose(1,0)) + 1e-30
                    p_R_test = density_data(test_set.transpose(1,0)) + 1e-30

                if args.density_estimator == "bnaf":
                    log_p_rel = np.concatenate([p_G_train-p_R_train, p_G_test-p_R_test])
                elif args.density_estimator == "kde":
                    log_p_rel = np.concatenate([p_G_train/p_R_train, p_G_test/p_R_test])

                thres = np.quantile(log_p_rel, 0.5)
                auc_y = np.hstack((np.array([1]*training_set.shape[0]), np.array([0]*test_set.shape[0])))
                fpr, tpr, thresholds = metrics.roc_curve(auc_y, log_p_rel, pos_label=1)
                auc = metrics.auc(fpr, tpr)
                if args.density_estimator == "bnaf":
                    print('Eqn.(2), training set prediction acc', (p_G_train-p_R_train >= thres).sum(0) / SIZE_PARAM)
                    print('Eqn.(2), AUC', auc)
                    performance_logger[f'{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}'][f'{N_DATA_GEN}_Eqn2'] = (p_G_train-p_R_train > thres).sum(0) / SIZE_PARAM
                elif args.density_estimator == "kde":
                    print('Eqn.(2), training set prediction acc', (p_G_train/p_R_train >= thres).sum(0) / SIZE_PARAM)
                    print('Eqn.(2), AUC', auc)
                    performance_logger[f'{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}'][f'{N_DATA_GEN}_Eqn2'] = (p_G_train/p_R_train > thres).sum(0) / SIZE_PARAM
                #print('Eqn.(2), test set prediction acc', (p_G_test-p_R_test > thres).sum(0) / SIZE_PARAM)

                performance_logger[f'{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}'][f'{N_DATA_GEN}_Eqn2AUC'] = auc
                performance_logger[f'{SIZE_PARAM}_{TRAINING_EPOCH}_{ADDITION_SIZE}'][f'{N_DATA_GEN}_Eqn2Score'] = log_p_rel


                np.save(f'results_folder/CELEBA_{PATH_CELEB_REPRESENTATION}_{args.batch_dim}_{args.hidden_dim}_{args.layers}_{args.epochs}_{args.gan_method}_{args.density_estimator}_{args.dataset}_trn_sz{args.training_size_list}_ref_sz{args.held_out_size_list}_gen_sz{args.gen_size_list}_{args.seed}.npy',performance_logger)
                
