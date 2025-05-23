import torch 
import torch.nn as nn
import matplotlib.pyplot as plt

def save_image(x, filename, figsize=(10, 10)):
    fig, axes = plt.subplots(4, 4, figsize=figsize)
    
    for i, ax in enumerate(axes.flat):
        ax.plot(x[i])
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=500, bbox_inches='tight')
    plt.close()

class Generator(nn.Module):
    def __init__(
        self,
        latent_dim=64,
        n_classes=2,
        lin_num_neurons=[512*3*12],
        lin_activation=nn.LeakyReLU(0.2),
        deconv_activation=nn.LeakyReLU(0.2),
        deconv_kernel_size=5,
        deconv_stride=2,
        deconv_padding=2,
        deconv_filters=[128, 64, 32, 16, 8],
        output_shape=800,
    ):
        super(Generator, self).__init__()
        
        self.lin_gen_class = nn.Sequential(*[
            nn.Linear(
                    in_features=n_classes, 
                    out_features=latent_dim,
                ),
                lin_activation,
        ])
        
        lin_layers = [
            nn.Linear(
                in_features=latent_dim*2,
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
        
        self.lin_gen = nn.Sequential(*lin_layers)

        self.initial_length = 12
        self.initial_channels = lin_num_neurons[-1] // self.initial_length
        
        deconv_layers = [
            nn.ConvTranspose1d(
                in_channels=self.initial_channels, 
                out_channels=deconv_filters[0],
                kernel_size=deconv_kernel_size,
                stride=deconv_stride,
                padding=deconv_padding,
                ),
            nn.BatchNorm1d(deconv_filters[0]),
            deconv_activation,
        ]
        
        for i in range(len(deconv_filters)):
            if i == (len(deconv_filters) - 1):
                deconv_layers.extend([
                    nn.ConvTranspose1d(
                        in_channels=deconv_filters[-1], 
                        out_channels=1,
                        kernel_size=deconv_kernel_size,
                        stride=deconv_stride,
                        padding=deconv_padding,
                    ),
                ])
                
            else:
                deconv_layers.extend([
                    nn.ConvTranspose1d(
                        in_channels=deconv_filters[i], 
                        out_channels=deconv_filters[i + 1],
                        kernel_size=deconv_kernel_size,
                        stride=deconv_stride,
                        padding=deconv_padding,
                    ),
                    nn.BatchNorm1d(deconv_filters[i + 1]),
                    deconv_activation,
                ])
                                
        self.deconv_gen = nn.Sequential(*deconv_layers)
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=output_shape)
        
    def forward(self, z, y):
        y_out = self.lin_gen_class(y)
        lin_in = torch.concatenate([z, y_out], dim=1)
        lin_out = self.lin_gen(lin_in)
        deconv_in = lin_out.reshape(-1, self.initial_channels, self.initial_length)
        deconv_out = self.deconv_gen(deconv_in)
        return self.adaptive_pool(deconv_out)

class Discriminator(nn.Module):
    def __init__(
        self,
        n_classes=2,
        cv_channels=[1, 4],
        cv_kernel=5,
        cv_stride=2,
        cv_padding=2,
        cv_activation=nn.LeakyReLU(0.2),
        lin_num_neurons=[32],
        lin_activation=nn.LeakyReLU(0.2),
        input_shape=800,
    ):
        super(Discriminator, self).__init__()
        
        cv_layers = []
        for i in range(len(cv_channels) - 1):
            cv_layers.extend([
                nn.Conv1d(
                    in_channels=cv_channels[i], 
                    out_channels=cv_channels[i + 1],
                    kernel_size=cv_kernel,
                    stride=cv_stride,
                    padding=cv_padding,
                ),
                nn.BatchNorm1d(cv_channels[i + 1]),
                cv_activation,
            ])
                
        self.cv_dis = nn.Sequential(*cv_layers)
        
        with torch.no_grad():
            dummy = torch.randn(1, 1, input_shape)
            conv_output_shape = self.cv_dis(dummy).squeeze().shape
            lin_input_neurons = self.cv_dis(dummy).flatten().shape[0]
                        
        self.lin_dis_curve = nn.Sequential(*[
            nn.Linear(
                    in_features=lin_input_neurons, 
                    out_features=int(lin_num_neurons[0] / 2),
                ),
                lin_activation,
        ])
        
        self.lin_dis_class = nn.Sequential(*[
            nn.Linear(
                    in_features=n_classes, 
                    out_features=int(lin_num_neurons[0] / 2),
                ),
                lin_activation,
        ])
        
        lin_layers = []
        
        for i in range(len(lin_num_neurons)):
            if i == len(lin_num_neurons) - 1:
                lin_layers.extend([
                    nn.Linear(
                        in_features=lin_num_neurons[-1], 
                        out_features=1,
                    ),
                    nn.Sigmoid(),
                ])
            else:
                lin_layers.extend([
                    nn.Linear(
                        in_features=lin_num_neurons[i], 
                        out_features=lin_num_neurons[i + 1],
                    ),
                    lin_activation,
                ])
        
        self.lin_dis = nn.Sequential(*lin_layers)
            
    def forward(self, x, y):
        cv_out = self.cv_dis(x)
        cv_out = cv_out.view(cv_out.size(0), -1)
        y_out = self.lin_dis_class(y)
        x_out = self.lin_dis_curve(cv_out)
        lin_in = torch.concatenate([x_out, y_out], dim=1)
        lin_out = self.lin_dis(lin_in)
        return lin_out
    
class cGAN(nn.Module):
    def __init__(
        self,
        device='cpu',
        latent_dim=64,
        lin_activation=nn.LeakyReLU(0.2),
        cv_activation=nn.LeakyReLU(0.2),
        lin_num_neurons_generator=[512*3*12],
        deconv_kernel_size_generator=5,
        deconv_padding_generator=2,
        deconv_stride_generator=2,
        deconv_filters_generator=[128, 64, 32, 16, 8],
        cv_channels_discriminator=[1, 4],
        cv_kernel_discriminator=5,
        cv_stride_discriminator=2,
        cv_padding_discriminator=2,
        lin_num_neurons_discriminator=[32],
        n_classes=2,
        shape=800,
    ):
        super(cGAN, self).__init__()
        
        self.device = device
        self.latent_dim = latent_dim
        
        self.generator = Generator(
            latent_dim=latent_dim,
            n_classes=n_classes,
            lin_num_neurons=lin_num_neurons_generator,
            lin_activation=lin_activation,
            deconv_activation=cv_activation,
            deconv_kernel_size=deconv_kernel_size_generator,
            deconv_stride=deconv_stride_generator,
            deconv_padding=deconv_padding_generator,
            deconv_filters=deconv_filters_generator,
            output_shape=shape,
        )
        
        self.discriminator = Discriminator(
            n_classes=n_classes,
            cv_channels=cv_channels_discriminator,
            cv_kernel=cv_kernel_discriminator,
            cv_stride=cv_stride_discriminator,
            cv_padding=cv_padding_discriminator,
            cv_activation=cv_activation,
            lin_num_neurons=lin_num_neurons_discriminator,
            lin_activation=lin_activation,
            input_shape=shape,
        )

    def latent_space(self, batch_size):
        return torch.randn(batch_size, self.latent_dim)
    
    def fit(
        self,
        data_loader,
        optimizer_generator,
        optimizer_discriminator,
        criterion,
        model_path=f'best_models/cgan',
        epochs=200,
        label_smoothing=0.8,
        verbose=True,
    ):
        self.to(self.device)

        generator_losses = []
        discriminator_losses = []
        discriminator_fake_acc = []
        discriminator_real_acc = []
        x_curve = torch.randn(16, self.latent_dim, device=self.device)
        y_curve = torch.zeros(16, 2, device=self.device)
        y_curve[:8, 0] = 1
        y_curve[8:, 1] = 1

        for epoch in range(epochs):
            epoch_generator_loss = 0.0
            epoch_discriminator_loss = 0.0
            epoch_real_accuracy = 0.0
            epoch_fake_accuracy = 0.0
            num_batches = 0

            for x_batch, y_batch in data_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                batch_size = x_batch.size(0)

                real_labels = torch.ones(batch_size, 1, device=self.device) * label_smoothing
                fake_labels = torch.zeros(batch_size, 1, device=self.device) + (1 - label_smoothing) 

                # Discriminator
                optimizer_discriminator.zero_grad()

                real_preds = self.discriminator(x_batch, y_batch)
                real_loss = criterion(real_preds, real_labels)

                z = self.latent_space(batch_size=batch_size).to(self.device)
                fake_batch = self.generator(z, y_batch).detach()
                fake_preds = self.discriminator(fake_batch, y_batch)
                fake_loss = criterion(fake_preds, fake_labels)

                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_discriminator.step()

                # Generator
                optimizer_generator.zero_grad()
                z = self.latent_space(batch_size=batch_size).to(self.device)
                fake_batch = self.generator(z, y_batch)
                g_loss = criterion(self.discriminator(fake_batch, y_batch), real_labels)  
                g_loss.backward()
                optimizer_generator.step()

                with torch.no_grad():
                    real_accuracy = (real_preds > 0.5).float().mean().item()
                    fake_accuracy = (fake_preds < 0.5).float().mean().item()
                    epoch_real_accuracy += real_accuracy
                    epoch_fake_accuracy += fake_accuracy

                epoch_generator_loss += g_loss.item()
                epoch_discriminator_loss += d_loss.item()
                num_batches += 1

            avg_generator_loss = epoch_generator_loss / num_batches
            avg_discriminator_loss = epoch_discriminator_loss / num_batches
            avg_real_accuracy = (epoch_real_accuracy / num_batches) * 100  
            avg_fake_accuracy = (epoch_fake_accuracy / num_batches) * 100

            generator_losses.append(avg_generator_loss)
            discriminator_losses.append(avg_discriminator_loss)
            discriminator_fake_acc.append(avg_fake_accuracy)
            discriminator_real_acc.append(avg_real_accuracy)

            if (epoch % 10) == 9:
                torch.save({
                    'generator': self.generator.state_dict(),
                    'discriminator': self.discriminator.state_dict(),
                    'epoch': epoch,
                }, f'{model_path}_epoch{epoch + 1}.pth')
                
                with torch.no_grad():
                    self.generator.eval()
                    curve = self.generator(x_curve, y_curve).cpu().reshape(16, -1)
                    save_image(curve, f'images/images_epoch{epoch + 1}.png')
                self.generator.train()

            if verbose:
                print(f'Epoch {epoch+1}/{epochs} | G_loss: {avg_generator_loss:.5f} | D_loss: {avg_discriminator_loss:.5f} | D_real_accuracy: {avg_real_accuracy:.5f} | D_fake_accuracy: {avg_fake_accuracy:.5f}')

        self.generator_losses = generator_losses
        self.discriminator_losses = discriminator_losses
        self.discriminator_fake_acc = discriminator_fake_acc
        self.discriminator_real_acc = discriminator_real_acc
