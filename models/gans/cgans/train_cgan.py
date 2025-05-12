import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision

import timeit
import argparse
import numpy as np
import itertools as it

from cgan import cGAN

# Instatiating model

parser = argparse.ArgumentParser()
parser.add_argument("--ld", type=int, default=64)
args = parser.parse_args()
latent_dim = args.ld
print(f'LD={latent_dim}')

datasets = 'ligcl_ausio2' # fake_real
n_classes = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using: {device}')

cgan = cGAN(
    device=device,
    latent_dim=latent_dim,
    n_classes=n_classes,
)

# Loading and transforming real data

batch_size = 64

#data1 = np.load('../tipAu_Au_morl.npz')['wavelets']
data2 = np.load('../tipAu_SiO2_morl.npz')['wavelets']
data3 = np.load('../tipLig_CL_morl.npz')['wavelets']
#data4 = np.load('../tipLig_SiO2_morl.npz')['wavelets']

#data1 = torch.tensor(data1, dtype=torch.float32).unsqueeze(1)
data2 = torch.tensor(data2, dtype=torch.float32).unsqueeze(1)
data3 = torch.tensor(data3, dtype=torch.float32).unsqueeze(1)
#data4 = torch.tensor(data4, dtype=torch.float32).unsqueeze(1)

#data_auau = data1 / data1.max()
data_ausio2 = data2 / data2.max()
data_ligcl = data3 / data3.max()
#data_ligsio2 = data4 / data4.max()

labels0 = torch.zeros(data_ausio2.shape[0], n_classes, dtype=torch.float32)
labels1 = torch.zeros(data_ausio2.shape[0], n_classes, dtype=torch.float32)

labels0[:, 0] = 1
labels1[:, 1] = 1

#data = torch.concatenate([data_auau, data_ausio2, data_ligcl, data_ligsio2])
data = torch.concatenate([data_ausio2, data_ligcl])
labels = torch.concatenate([labels1, labels0])

dataset = TensorDataset(data, labels)

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Setting GANs loss and optimizers

epochs, lr_g, lr_d = 400, 5e-4, 2e-4

optimizer_G = torch.optim.Adam(cgan.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(cgan.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# Training

start = timeit.default_timer()
                    
cgan.fit(
    data_loader=data_loader,
    optimizer_generator=optimizer_G,
    optimizer_discriminator=optimizer_D,
    criterion=criterion,
    model_path=f'best_models/cgan_ld{latent_dim}_{datasets}',
    epochs=epochs,
    verbose=True,
)
    
end = timeit.default_timer()

print(f'{int(end-start)} seconds elapsed.')
print()

loss_g = np.array(cgan.generator_losses)
loss_d = np.array(cgan.discriminator_losses)
fake_acc, real_acc = cgan.discriminator_fake_acc, cgan.discriminator_real_acc
np.save(f'loss_cgan_ld{latent_dim}_g_{datasets}.npy', loss_g)
np.save(f'loss_cgan_ld{latent_dim}_d_{datasets}.npy', loss_d)
np.save(f'loss_cgan_ld{latent_dim}_d_fake_{datasets}.npy', fake_acc)
np.save(f'loss_cgan_ld{latent_dim}_d_real_{datasets}.npy', real_acc)
