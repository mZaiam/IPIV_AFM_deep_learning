import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision

from cnn import CNN

# Instatiating model

out = 16
cv_out = 64
num_classes = 10

cv_channels = [1, 32, cv_out]
cv_kernels = [3, 3]
cv_activation = nn.ReLU()
cv_dropout = [0, 0.25]
pool_size = [24, out]
lin_neurons = [cv_out * out**2, 128, num_classes]
lin_activation = nn.ReLU()
lin_dropout = [0.25, 0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Using: {device}')

cnn = CNN(
    cv_channels,
    cv_kernels,
    cv_activation,
    cv_dropout,
    pool_size,
    lin_neurons,
    lin_activation,
    lin_dropout,
    device
)

# Loading and transforming data

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
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

optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

cnn.fit(loader_train, loader_test, optimizer, criterion, epochs=epochs)
cnn.plot_learning_curves('images/cnn_curve')

# Performance

y_pred = torch.max(cnn.predict(loader_test), dim=1)[1]
y_test = test_data.targets.clone().detach()
y_pred, y_test = y_pred.cpu(), y_test.cpu() 
accuracy = torch.sum(y_pred == y_test).item() / y_test.size(0)

print(f'Accuracy: {accuracy}')
