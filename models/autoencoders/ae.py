import torch 
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(
        self,
        conv_num_layers=3,
        conv_activation=nn.ReLU(),
        conv_kernel_size=5,
        conv_filters=[128, 64, 32],
        pool_kernel_size=3,
        lin_num_layers=3,
        lin_num_neurons=[128, 32],
        lin_activation=nn.ReLU(),
        input_shape=(127, 800),
        latent_dim=2,
    ):
        super(Encoder, self).__init__()
        
        conv_layers = [
            nn.Conv2d(
                    in_channels=1, 
                    out_channels=conv_filters[0],
                    kernel_size=conv_kernel_size,
                ),
                conv_activation,
                nn.AvgPool2d(
                    kernel_size=pool_kernel_size,
                ),
        ]
        
        for i in range(conv_num_layers - 1):
            conv_layers.extend([
                nn.Conv2d(
                    in_channels=conv_filters[i], 
                    out_channels=conv_filters[i + 1],
                    kernel_size=conv_kernel_size,
                ),
                conv_activation,
                nn.AvgPool2d(
                    kernel_size=pool_kernel_size,
                ),
            ])
                
        self.conv_enc = nn.Sequential(*conv_layers)
        
        with torch.no_grad():
            dummy = torch.randn(1, 1, *input_shape)
            conv_output_shape = self.conv_enc(dummy).squeeze().shape
            lin_input_neurons = self.conv_enc(dummy).flatten().shape[0]
            
        self.conv_output_shape = conv_output_shape    
        self.lin_input_neurons = lin_input_neurons
        
        lin_layers = [
            nn.Linear(
                    in_features=self.lin_input_neurons, 
                    out_features=lin_num_neurons[0],
                ),
                lin_activation,
        ]
        
        for i in range(lin_num_layers - 1):
            if i == (lin_num_layers - 2):
                lin_layers.extend([
                    nn.Linear(
                        in_features=lin_num_neurons[-1], 
                        out_features=latent_dim,
                    ),
                ])
            
            else:
                lin_layers.extend([
                    nn.Linear(
                        in_features=lin_num_neurons[i], 
                        out_features=lin_num_neurons[i + 1],
                    ),
                    lin_activation,
                ])
        
        self.lin_enc = nn.Sequential(*lin_layers)
        
    def forward(self, x):
        conv_out = self.conv_enc(x)
        lin_in = conv_out.reshape(conv_out.shape[0], -1)
        lin_out = self.lin_enc(lin_in)
        return lin_out
    
class Decoder(nn.Module):
    def __init__(
        self,
        lin_output_neurons,
        conv_input_shape,
        latent_dim=2,
        lin_num_layers=2,
        lin_num_neurons=[32, 128],
        lin_activation=nn.ReLU(),
        deconv_num_layers=3,
        deconv_activation=nn.ReLU(),
        deconv_kernel_size=7,
        deconv_stride=5,
        deconv_filters=[64, 128],
        output_shape=(127, 800),
    ):
        super(Decoder, self).__init__()
        
        self.conv_input_shape = conv_input_shape
        
        lin_layers = [
            nn.Linear(
                    in_features=latent_dim, 
                    out_features=lin_num_neurons[0],
                ),
                lin_activation,
        ]
        
        for i in range(lin_num_layers):
            if i == (lin_num_layers - 1):
                lin_layers.extend([
                    nn.Linear(
                        in_features=lin_num_neurons[-1], 
                        out_features=lin_output_neurons,
                    ),
                    lin_activation,
                ])
                
            else:
                lin_layers.extend([
                    nn.Linear(
                        in_features=lin_num_neurons[i], 
                        out_features=lin_num_neurons[i + 1],
                    ),
                    lin_activation,
                ])
        
        self.lin_dec = nn.Sequential(*lin_layers)
        
        deconv_layers = [
            nn.ConvTranspose2d(
                    in_channels=conv_input_shape[0], 
                    out_channels=deconv_filters[0],
                    kernel_size=deconv_kernel_size,
                ),
                deconv_activation,
        ]
        
        for i in range(deconv_num_layers - 1):
            if i == (deconv_num_layers - 2):
                deconv_layers.extend([
                    nn.ConvTranspose2d(
                        in_channels=deconv_filters[-1], 
                        out_channels=1,
                        kernel_size=deconv_kernel_size,
                        stride=deconv_stride,
                    )
                ])
                
            else:
                deconv_layers.extend([
                    nn.ConvTranspose2d(
                        in_channels=deconv_filters[i], 
                        out_channels=deconv_filters[i + 1],
                        kernel_size=deconv_kernel_size,
                        stride=deconv_stride,
                    ),
                    deconv_activation,
                ])
                
        self.deconv_dec = nn.Sequential(*deconv_layers)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=output_shape)
        
    def forward(self, x):
        lin_out = self.lin_dec(x)
        deconv_in = lin_out.reshape(-1, *self.conv_input_shape)
        deconv_out = self.deconv_dec(deconv_in)
        return self.adaptive_pool(deconv_out)
    
class AE(nn.Module):
    def __init__(
        self,
        latent_dim=2,
        lin_activation=nn.ReLU(),
        lin_num_layers=3,
        lin_num_neurons=[128, 32],
        conv_activation=nn.ReLU(),
        conv_num_layers=3,
        conv_kernel_size=5,
        conv_filters=[128, 64, 32],
        pool_kernel_size=3,
        deconv_filters=[64, 128],
        deconv_kernel_size=7,
        deconv_stride=5,
        image_shape=(127, 800),
        device='cpu',
    ):
        super(AE, self).__init__()
        
        self.device = device
        
        self.encoder = Encoder(
            conv_num_layers=conv_num_layers,
            conv_activation=conv_activation,
            conv_kernel_size=conv_kernel_size,
            conv_filters=conv_filters,
            pool_kernel_size=pool_kernel_size,
            lin_num_layers=lin_num_layers,
            lin_num_neurons=lin_num_neurons,
            lin_activation=lin_activation,
            input_shape=image_shape,
            latent_dim=latent_dim,
        )
                
        self.decoder = Decoder(
            lin_output_neurons=self.encoder.lin_input_neurons,
            conv_input_shape=self.encoder.conv_output_shape,
            latent_dim=latent_dim,
            lin_num_layers=(lin_num_layers - 1),
            lin_num_neurons=list(reversed(lin_num_neurons)),
            lin_activation=lin_activation,
            deconv_num_layers=conv_num_layers,
            deconv_activation=conv_activation,
            deconv_kernel_size=deconv_kernel_size,
            deconv_stride=deconv_stride,
            deconv_filters=deconv_filters,
            output_shape=image_shape,
        )
        
    def forward(self, x):
        enc_out = self.encoder(x)
        dec_out = self.decoder(enc_out)
        return dec_out
    
    def fit(
        self, 
        loader_data, 
        optimizer, 
        criterion, 
        epochs=200, 
        model_path='best_models/autoencoder.pth',
        patience=20,  
        verbose=True,
    ):
        self.to(self.device)

        self.epochs = []
        losses = []
        best_loss = float('inf')
        counter = 0  

        for epoch in range(epochs):
            self.train()  
            loss_epoch = 0.0

            for x_batch in loader_data:
                x_batch = x_batch.to(self.device)

                optimizer.zero_grad()  
                x_pred = self.forward(x_batch)  
                loss = criterion(x_pred, x_batch)  
                loss.backward() 
                optimizer.step() 

                loss_epoch += loss.item()

            loss_epoch /= len(loader_data)
            losses.append(loss_epoch)

            self.epochs.append(epoch + 1)

            if verbose:
                print(f'Epoch {epoch+1}/{epochs} | Loss: {loss_epoch:.10f}')
                
            if loss_epoch < best_loss:
                best_loss = loss_epoch  
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

        self.losses = losses
        self.model_path = model_path
