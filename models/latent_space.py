import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision

import timeit
import argparse
import numpy as np

from ae import AE

def calculate_rmse(model, dataloader, device):
    model.eval()
    total_mse = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for x_batch in dataloader:
            x_batch = x_batch.to(device)
            x_recon = model(x_batch)
            mse = nn.MSELoss(reduction='sum')(x_recon, x_batch)
            total_mse += mse.item()
            total_samples += x_batch.size(0)
    
    rmse = np.sqrt(total_mse / total_samples)
    return rmse

# Instatiating model

parser = argparse.ArgumentParser()
parser.add_argument("--ld", type=int, default=2)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
latent_dim = args.ld
print(f'LD={latent_dim}')
datasets = 'all'

print(f'Using: {device}')

ae = AE(
    device=device,
    latent_dim=latent_dim,
)

# Loading and transforming data

data1 = np.load('tipAu_Au_morl.npz')['wavelets']
data2 = np.load('tipAu_SiO2_morl.npz')['wavelets']
data3 = np.load('tipLig_CL_morl.npz')['wavelets']
data4 = np.load('tipLig_SiO2_morl.npz')['wavelets']

data1 = torch.tensor(data1, dtype=torch.float32).unsqueeze(1)
data2 = torch.tensor(data2, dtype=torch.float32).unsqueeze(1)
data3 = torch.tensor(data3, dtype=torch.float32).unsqueeze(1)
data4 = torch.tensor(data4, dtype=torch.float32).unsqueeze(1)

data_auau = data1 / data1.max()
data_ausio2 = data2 / data2.max()
data_ligcl = data3 / data3.max()
data_ligsio2 = data4 / data4.max()

data = torch.concatenate([data_auau, data_ausio2, data_ligcl, data_ligsio2])

# Latent Space

model = torch.load(f'best_models/autoencoder_ld{latent_dim}_{datasets}.pth', map_location=device, weights_only=True)
ae.load_state_dict(model)
ae.to(device)
ae.eval()

loaders = {
    'auau': DataLoader(data_auau, batch_size=64, shuffle=False),
    'ausio2': DataLoader(data_ausio2, batch_size=64, shuffle=False),
    'ligcl': DataLoader(data_ligcl, batch_size=64, shuffle=False),
    'ligsio2': DataLoader(data_ligsio2, batch_size=64, shuffle=False),
    'all': DataLoader(data, batch_size=64, shuffle=False)
}

ld_auau = []

for x_batch in loaders['auau']:
    with torch.no_grad():
        x_pred = ae.encoder(x_batch.to(device))
        ld_auau.append(x_pred)

ld_auau = torch.concatenate(ld_auau)

ld_ausio2 = []

for x_batch in loaders['ausio2']:
    with torch.no_grad():
        x_pred = ae.encoder(x_batch.to(device))
        ld_ausio2.append(x_pred)

ld_ausio2 = torch.concatenate(ld_ausio2)

ld_ligcl = []

for x_batch in loaders['ligcl']:
    with torch.no_grad():
        x_pred = ae.encoder(x_batch.to(device))
        ld_ligcl.append(x_pred)

ld_ligcl = torch.concatenate(ld_ligcl)

ld_ligsio2 = []

for x_batch in loaders['ligsio2']:
    with torch.no_grad():
        x_pred = ae.encoder(x_batch.to(device))
        ld_ligsio2.append(x_pred)

ld_ligsio2 = torch.concatenate(ld_ligsio2)

np.save(f'ld{latent_dim}_auau_{datasets}.npy', ld_auau.cpu().numpy())
np.save(f'ld{latent_dim}_ausio2_{datasets}.npy', ld_ausio2.cpu().numpy())
np.save(f'ld{latent_dim}_ligcl_{datasets}.npy', ld_ligcl.cpu().numpy())
np.save(f'ld{latent_dim}_ligsio2_{datasets}.npy', ld_ligsio2.cpu().numpy())

# RMSE

print('RMSE')
for name, loader in loaders.items():
    rmse = calculate_rmse(ae, loader, device)
    print(f"{name}: {rmse:.4f}")
