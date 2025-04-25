import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

import argparse
import numpy as np

from mlp import NN

# Instatiating model

parser = argparse.ArgumentParser()
parser.add_argument("--ld", type=int, default=2)
args = parser.parse_args()

ld = args.ld
print(f'LD={ld}')
num_classes = 4
test_split = 0.8

size_layers = [ld, 32, 32, 32, num_classes]
dropout_layers = [0, 0, 0, 0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Using: {device}')

mlp = NN(size_layers, dropout_layers, nn.ReLU(), device)

# Loading data

data_auau = torch.tensor(np.load(f"ld{ld}_auau_all.npy"), dtype=torch.float32)
data_ausio2 = torch.tensor(np.load(f"ld{ld}_ausio2_all.npy"), dtype=torch.float32)
data_ligcl = torch.tensor(np.load(f"ld{ld}_ligcl_all.npy"), dtype=torch.float32)
data_ligsio2 = torch.tensor(np.load(f"ld{ld}_ligsio2_all.npy"), dtype=torch.float32)

labels_auau = torch.zeros(len(data_auau), dtype=torch.long)
labels_ausio2 = torch.ones(len(data_ausio2), dtype=torch.long)
labels_ligcl = torch.full((len(data_ligcl),), 2, dtype=torch.long)
labels_ligsio2 = torch.full((len(data_ligsio2),), 3, dtype=torch.long)

data = torch.cat([data_auau, data_ausio2, data_ligcl, data_ligsio2], dim=0)
labels = torch.cat([labels_auau, labels_ausio2, labels_ligcl, labels_ligsio2], dim=0)

dataset = TensorDataset(data, labels)

total_size = len(dataset)
train_size = int(test_split * total_size)
test_size = total_size - train_size

train_dataset, test_dataset = random_split(
    dataset, 
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42) 
)

batch_size = 32 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def check_proportions(loader):
    counts = torch.zeros(4)
    for _, labels in loader:
        for i in range(4):
            counts[i] += (labels == i).sum()
    return counts / counts.sum()

print("Train proportions:", check_proportions(train_loader))
print("Test proportions:", check_proportions(test_loader))

# Training

batch_size, epochs, lr, patience = 64, 500, 1e-3, 50

optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

mlp.fit(
    train_loader, 
    test_loader, 
    optimizer, 
    criterion, 
    patience=patience,
    epochs=epochs,
    model_path=f"best_models/mlp_ld{ld}.pth"
)

loss_train = np.array(mlp.loss_train)
loss_test = np.array(mlp.loss_val)

np.save(f'loss_train_mlp_ld{ld}.npy', loss_train)
np.save(f'loss_test_mlp_ld{ld}.npy', loss_test)

# Performance

model = torch.load(f'best_models/mlp_ld{ld}.pth', map_location=device, weights_only=True)
mlp.load_state_dict(model)
mlp.to(device)
mlp.eval()  

all_preds = []
all_labels = []

with torch.no_grad():  
    for inputs, labels in test_loader:
        outputs = mlp(inputs.to(device))  
        _, preds = torch.max(outputs, 1)  
        all_preds.append(preds)
        all_labels.append(labels)

y_pred = torch.cat(all_preds)
y_test = torch.cat(all_labels)
y_pred, y_test = y_pred.cpu(), y_test.cpu()
accuracy = (y_pred == y_test).float().mean().item()

print(f'Test Accuracy: {accuracy * 100:.2f}%')
