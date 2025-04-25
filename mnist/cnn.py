import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(
        self,
        cv_channels,
        cv_kernels,
        cv_activation,
        cv_dropout,
        pool_size,
        lin_neurons,
        lin_activation,
        lin_dropout,
        device,
    ):
        '''Convolutional NN.
        
        Args:
            cv_channels:    list of channel sizes for convolutional layers.
            cv_kernels:     list of kernel sizes for convolutional layers.
            cv_activation:  torch activation function.
            cv_dropout:     list of dropout rates for convolutional layers.
            pool_size:      list of output sizes for adaptive pooling layers.
            lin_neurons:    list of neuron counts for linear layers.
            lin_activation: torch activation function.
            lin_dropout:    list of dropout rates for linear layers.
            device:         torch device.
        '''
        super(CNN, self).__init__()
        
        cv_layers = []
        for i in range(len(cv_channels) - 1):
            cv_layers.extend([
                nn.Conv2d(
                    in_channels=cv_channels[i], 
                    out_channels=cv_channels[i + 1],
                    kernel_size=cv_kernels[i],
                ),
                cv_activation,
                nn.AdaptiveAvgPool2d(
                    output_size=pool_size[i]
                ),
            ])
            
            if cv_dropout[i] != 0:
                cv_layers.append(
                    nn.Dropout(p=cv_dropout[i])
                )
                
        self.cv = nn.Sequential(*cv_layers)
            
        lin_layers = []
        for i in range(len(lin_neurons) - 1):
            lin_layers.extend([
                nn.Linear(
                    in_features=lin_neurons[i], 
                    out_features=lin_neurons[i + 1],
                ),
                lin_activation,
            ])
            if lin_dropout[i] != 0:
                lin_layers.append(nn.Dropout(p=lin_dropout[i]))
        
        self.lin = nn.Sequential(*lin_layers)
        
        self.device = device
            
    def forward(self, x):
        cv_out = self.cv(x)
        cv_out = cv_out.view(cv_out.size(0), -1)
        lin_out = self.lin(cv_out)
        return lin_out
    
    def fit(
        self, 
        loader_train, 
        loader_val, 
        optimizer, 
        criterion, 
        epochs=100, 
        model_path='models/cnn.pth',
        patience=20,  
        verbose=True,
    ):
        self.to(self.device)

        self.epochs = []
        loss_train = []
        loss_val = []
        best_loss = float('inf')
        counter = 0  

        for epoch in range(epochs):
            self.train()  
            loss_train_epoch = 0.0

            for x_batch, y_batch in loader_train:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()  
                y_pred = self.forward(x_batch)  
                loss = criterion(y_pred, y_batch)  
                loss.backward() 
                optimizer.step() 

                loss_train_epoch += loss.item()

            loss_train_epoch /= len(loader_train)
            loss_train.append(loss_train_epoch)

            self.eval()  
            loss_val_epoch = 0.0

            with torch.no_grad():
                for x_batch, y_batch in loader_val:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    y_pred = self.forward(x_batch) 
                    loss = criterion(y_pred, y_batch) 
                    loss_val_epoch += loss.item()

            loss_val_epoch /= len(loader_val)
            loss_val.append(loss_val_epoch)

            self.epochs.append(epoch + 1)

            if verbose:
                print(f'Epoch {epoch+1}/{epochs}: train_loss: {loss_train_epoch:.4f} val_loss {loss_val_epoch:.4f}')
                
            if loss_val_epoch < best_loss:
                best_loss = loss_val_epoch  
                counter = 0  
                torch.save(self.state_dict(), model_path)  
                if verbose:
                    print(f'Saved at epoch {epoch + 1}.')
            else:
                counter += 1

            if counter >= patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch+1}.')
                break

        self.loss_train = loss_train
        self.loss_val = loss_val
        self.model_path = model_path
    
    def predict(self, loader_test):
        if hasattr(self, 'model_path'):
            self.load_state_dict(torch.load(self.model_path, weights_only=True))
            self.to(self.device)

        self.eval() 
        y = []

        with torch.no_grad():
            for x_batch, _ in loader_test:
                x_batch = x_batch.to(self.device)
                y_pred = self.forward(x_batch)
                y.append(y_pred)

        return torch.cat(y, dim=0)        

    def plot_learning_curves(self, file_name):
        plt.plot(self.epochs, self.loss_train, color='blue', label='Train')
        plt.plot(self.epochs, self.loss_val, color='orange', label='Val')
        plt.legend()
        plt.title('Learning Curves')
        plt.savefig(file_name + '.png', bbox_inches='tight')
        plt.show()
