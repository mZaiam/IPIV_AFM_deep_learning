import torch 
import torch.nn as nn

class Generator(nn.Module):
    def __init__(
        self,
        latent_dim=2,
        lin_num_neurons=[32, 128, 1024],
        lin_activation=nn.LeakyReLU(0.2),
        deconv_activation=nn.LeakyReLU(0.2),
        deconv_kernel_size=7,
        deconv_stride=5,
        deconv_filters=[64, 128],
        output_shape=(127, 800),
    ):
        super(Generator, self).__init__()
        
        self.conv_input_shape = (1, 16, 64)
                
        lin_layers = [
            nn.Linear(
                    in_features=latent_dim, 
                    out_features=lin_num_neurons[0],
                ),
                lin_activation,
        ]
        
        for i in range(len(lin_num_neurons) - 1):
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
                    in_channels=self.conv_input_shape[0], 
                    out_channels=deconv_filters[0],
                    kernel_size=deconv_kernel_size,
                ),
                deconv_activation,
        ]
        
        for i in range(len(deconv_filters)):
            if i == (len(deconv_filters) - 1):
                deconv_layers.extend([
                    nn.ConvTranspose2d(
                        in_channels=deconv_filters[-1], 
                        out_channels=1,
                        kernel_size=deconv_kernel_size,
                        stride=deconv_stride,
                    ),
                    deconv_activation,
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
    
class Discriminator(nn.Module):
    def __init__(
        self,
        cv_channels=[1, 64, 32],
        cv_kernel=3,
        cv_activation=nn.LeakyReLU(0.2),
        pool_kernel=4,
        lin_num_neurons=[128, 32],
        lin_activation=nn.ReLU(0.2),
        n_classes=2,
        input_shape=(127, 800),
    ):
        super(Discriminator, self).__init__()
        
        cv_layers = []
        for i in range(len(cv_channels) - 1):
            cv_layers.extend([
                nn.Conv2d(
                    in_channels=cv_channels[i], 
                    out_channels=cv_channels[i + 1],
                    kernel_size=cv_kernel,
                ),
                cv_activation,
                nn.AvgPool2d(
                    kernel_size=pool_kernel
                ),
            ])
                
        self.cv = nn.Sequential(*cv_layers)
        
        with torch.no_grad():
            dummy = torch.randn(1, 1, *input_shape)
            conv_output_shape = self.cv(dummy).squeeze().shape
            lin_input_neurons = self.cv(dummy).flatten().shape[0]
                        
        lin_layers = [
            nn.Linear(
                    in_features=lin_input_neurons, 
                    out_features=lin_num_neurons[0],
                ),
                lin_activation,
        ]
        
        for i in range(len(lin_num_neurons)):
            if i == len(lin_num_neurons) - 1:
                lin_layers.extend([
                    nn.Linear(
                        in_features=lin_num_neurons[-1], 
                        out_features=n_classes,
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
        
        self.lin = nn.Sequential(*lin_layers)
            
    def forward(self, x):
        cv_out = self.cv(x)
        cv_out = cv_out.view(cv_out.size(0), -1)
        lin_out = self.lin(cv_out)
        return lin_out
    
class GAN(nn.Module):
    def __init__(
        self,
        device,
        latent_dim=2,
        lin_activation=nn.LeakyReLU(0.2),
        cv_activation=nn.LeakyReLU(0.2),
        lin_num_neurons_generator=[32, 128, 1024],
        deconv_kernel_size_generator=7,
        deconv_stride_generator=5,
        deconv_filters_generator=[64, 128],
        cv_channels_discriminator=[1, 64, 32],
        cv_kernel_discriminator=3,
        pool_kernel_discriminator=4,
        lin_num_neurons_discriminator=[128, 32],
        n_classes=2,
        shape=(127, 800),
    ):
        super(GAN, self).__init__()
        
        self.device = device
        self.latent_dim = latent_dim
        
        self.generator = Generator(
            latent_dim=latent_dim,
            lin_num_neurons=lin_num_neurons_generator,
            lin_activation=lin_activation,
            deconv_activation=cv_activation,
            deconv_kernel_size=deconv_kernel_size_generator,
            deconv_stride=deconv_stride_generator,
            deconv_filters=deconv_filters_generator,
            output_shape=shape,
            )
        
        self.discriminator = Discriminator(
            cv_channels=cv_channels_discriminator,
            cv_kernel=cv_kernel_discriminator,
            cv_activation=cv_activation,
            pool_kernel=pool_kernel_discriminator,
            lin_num_neurons=lin_num_neurons_discriminator,
            lin_activation=lin_activation,
            n_classes=n_classes,
            input_shape=shape,
        )

    def latent_space(self, batch_size=64):
        return torch.randn(batch_size, self.latent_dim)
    
    def fit(
        self,
        data_loader,
        optimizer_generator,
        optimizer_discriminator,
        criterion,
        model_path=f'best_models/gan.pth',
        batch_size=64,
        epochs=1000,
        p=2,
        verbose=True,
    ):
        self.to(self.device)
        
        best_generator_loss = float('inf')
        generator_losses = []
        discriminator_losses = []
        
        for epoch in range(epochs):
            epoch_generator_loss = 0.0
            epoch_discriminator_loss = 0.0
            num_batches = 0
            
            for real_batch in data_loader:
                for _ in range(p):
                    real_batch = real_batch.to(self.device)
                
                    # Optimizing generator
                    optimizer_generator.zero_grad()

                    fake_batch = self.generator(self.latent_space(batch_size=batch_size).to(self.device))
                    fake_labels = torch.zeros(batch_size, dtype=torch.long).to(self.device)
                    real_labels = torch.ones(batch_size, dtype=torch.long).to(self.device)

                    generator_loss = criterion(self.discriminator(fake_batch), real_labels)
                    generator_loss.backward()
                    optimizer_generator.step()
                    epoch_generator_loss += generator_loss.item() / p

                # Optimizing discriminator
                optimizer_discriminator.zero_grad()

                real_loss = criterion(self.discriminator(real_batch), real_labels)
                fake_loss = criterion(self.discriminator(fake_batch.detach()), fake_labels)
                discriminator_loss = (real_loss + fake_loss) / 2
                discriminator_loss.backward()
                optimizer_discriminator.step()
                
                epoch_discriminator_loss += discriminator_loss.item()
            
                num_batches += 1
            
            avg_generator_loss = epoch_generator_loss / num_batches
            avg_discriminator_loss = epoch_discriminator_loss / num_batches
            
            generator_losses.append(avg_generator_loss)
            discriminator_losses.append(avg_discriminator_loss)
        
            if avg_generator_loss < best_generator_loss:
                best_generator_loss = avg_generator_loss
                torch.save({
                    'generator': self.generator.state_dict(),
                    'discriminator': self.discriminator.state_dict(),
                    'epoch': epoch,
                }, model_path)

            if verbose:
                print(f'Epoch {epoch+1}/{epochs} | G_loss: {avg_generator_loss:.5f} | D_loss: {avg_discriminator_loss:.5f}')
        
        self.generator_losses = generator_losses
        self.discriminator_losses = discriminator_losses
