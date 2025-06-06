import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
from mlp import NN

# Instantiate model
parser = argparse.ArgumentParser()
parser.add_argument("--ld", type=int, default=2)
args = parser.parse_args()

ld = args.ld
print(f'LD={ld}')
num_classes = 2
test_size = 0.2 
random_seed = 42

size_layers = [ld, 32, num_classes]
dropout_layers = [0, 0, 0, 0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using: {device}')

mlp = NN(size_layers, dropout_layers, nn.ReLU(), device)
name = 'all'

# Loading data
data = torch.tensor(np.load(f"../../best_models/ld{ld}_ausio2_{name}.npy"), dtype=torch.float32)
labels = torch.tensor(np.load('../../dataset_TipAu_SiO2_annotated.npz')['labels'], dtype=torch.long).squeeze()

x_train, x_test, y_train, y_test = train_test_split(
    data.numpy(), labels.numpy(), 
    test_size=test_size, 
    random_state=random_seed,
    stratify=labels.numpy()
)
train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training

epochs, lr, patience = 500, 1e-3, 50

optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

mlp.fit(
    train_loader, 
    test_loader, 
    optimizer, 
    criterion, 
    patience=patience,
    epochs=epochs,
    model_path=f"best_models/mlp_ld{ld}_{name}.pth"
)

loss_train = np.array(mlp.loss_train)
loss_test = np.array(mlp.loss_val)

np.save(f'loss_train_mlp_ld{ld}_{name}.npy', loss_train)
np.save(f'loss_test_mlp_ld{ld}_{name}.npy', loss_test)

# Performance

model = torch.load(f'best_models/mlp_ld{ld}_{name}.pth', map_location=device, weights_only=True)
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
