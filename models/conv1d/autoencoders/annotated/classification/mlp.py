import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class NN(nn.Module):
    def __init__(
        self,
        size_layers,
        dropout_layers,
        activation,
        device,
    ):
        '''Linear NN.
        
        Args:
            size_layers:    list with number of neurons of each layer, including the input and output layers.
            dropout_layers: list with layers that contain dropout. Layers that contain dropout have a parameter 
                            greater than zero on the list. The first hidden layer is indexed at 0, and so on.
            activation:     torch function.
            device:         torch device.
        '''
        super(NN, self).__init__()
        
        layers = []
        
        for i in range(len(size_layers) - 1):
            layers.append(
                nn.Linear(in_features=size_layers[i], out_features=size_layers[i + 1]),
                
            )
            layers.append(activation)
            if dropout_layers[i] != 0:
                layers.append(
                    nn.Dropout(p=dropout_layers[i])
                )
            
        self.model = nn.Sequential(*layers)
        
        self.device = device
            
    def forward(self, x):
        return self.model(x)
    
    def fit(
        self, 
        loader_train, 
        loader_val, 
        optimizer, 
        criterion, 
        epochs=100, 
        model_path='models/mlp.pth',
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
