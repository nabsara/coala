import gin
import torch
import torch.nn as nn

from coala.models.blocks import TCN, CausalCNN, CausalCNN2

@gin.configurable
class CNNLatentEncoderWithExtraLatents(nn.Module):
    def __init__(self, latent_dim=32, n_control_dims=1, seq_length=256, hidden_dim=64, kernel_size=3, dilations=[[1, 2, 4], [1, 2, 4], [1, 2, 4]], control_dim=None, stream=False):
        super().__init__()

        self.latent_dim = latent_dim
        self.n_control_dims = n_control_dims
        self.input_dim = latent_dim + n_control_dims
        if control_dim is None:
            self.control_dim = self.input_dim
        else:
            self.control_dim = control_dim
            
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.stream = stream

        self.extra_latents = nn.Parameter(torch.randn(1, self.n_control_dims, self.seq_length), requires_grad=True)
        nn.init.normal_(self.extra_latents, mean=0.0, std=0.02)

        self.net = TCN(self.input_dim, hidden_dim, n_out=2, kernel_size=kernel_size, dilations=dilations, proj_dim=self.control_dim)

    def forward(self, z):

        extra = self.extra_latents.expand(z.shape[0], -1, -1)
        if self.stream:
            extra = extra[:,:,:z.shape[-1]]
        z_aug = torch.cat([extra, z], dim=1)

        out = self.net(z_aug)

        mu_c, logvar_c = out.chunk(2, 1)

        return mu_c, logvar_c 


@gin.configurable
class MiniCausalCNNLatentDecoder(nn.Module):
    def __init__(self, latent_dim=32, control_dim=None, seq_length=256, hidden_dim=64, kernel_size=1):
        super().__init__()

        self.latent_dim = latent_dim
        self.control_dim = latent_dim if control_dim is None else control_dim
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim

        self.net = CausalCNN(self.control_dim, hidden_dim, n_out=1, kernel_size=kernel_size, proj_dim=self.latent_dim)
    
    def forward(self, c):
        z = self.net(c)
        return z
    

@gin.configurable
class MiniCausalCNNLatentDecoder2(nn.Module):
    def __init__(self, latent_dim=32, control_dim=None, seq_length=256, hidden_dim=64):
        super().__init__()

        self.latent_dim = latent_dim
        self.control_dim = latent_dim if control_dim is None else control_dim
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim

        self.net = CausalCNN2(self.control_dim, hidden_dim, proj_dim=self.latent_dim)
    
    def forward(self, c):
        z = self.net(c)
        return z
