import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision

from mlp import NN

# Instatiating model

size_im = 28**2
num_classes = 10

size_layers = [size_im, 128, num_classes]
dropout_layers = [0.25, 0, 0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Using: {device}')

mlp = NN(size_layers, dropout_layers, nn.ReLU(), device)

# Loading and transforming data

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: x.view(-1)),
])

train_data = torchvision.datasets.MNIST(
    './data', 
    train=True,
    transform=transforms, 
    download=True,
)

test_data = torchvision.datasets.MNIST(
    './data', 
    train=False,
    transform=transforms, 
    download=True,
)

# Training

batch_size, epochs, lr = 64, 100, 1e-3

loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
loader_test = DataLoader(test_data, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

mlp.fit(loader_train, loader_test, optimizer, criterion, epochs=epochs)
mlp.plot_learning_curves('images/mlp_curve')

# Performance

y_pred = torch.max(mlp.predict(loader_test), 1)[1]
y_test = test_data.targets.clone().detach()
y_pred, y_test = y_pred.cpu(), y_test.cpu() 
accuracy = torch.sum(y_pred == y_test).item() / y_test.size(0)

print(f'Accuracy: {accuracy}')
