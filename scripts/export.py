import os
import gin
import pathlib
import click
import torch
import torch.nn as nn
import nn_tilde
import cached_conv as cc

from coala.models.model import CoALa

torch.set_grad_enabled(False)


class ScriptedCoALa(nn_tilde.Module):
    
    def __init__(
            self,
            pretrained: CoALa,
            ae_ratio: int,
        ):
        super().__init__()

        self.control_dim = pretrained.control_dim
        self.latent_size = pretrained.latent_size
        self.n_controls = pretrained.n_controls
        self.ae_ratio = ae_ratio

        self.encoder = pretrained.control_extractor
        self.decoder = pretrained.latent_projector

        self.register_method(
            "encode",
            in_channels=self.latent_size,
            in_ratio=self.ae_ratio,
            out_channels=self.control_dim,
            out_ratio=self.ae_ratio,
            input_labels=[f'(signal) Latent dimension {i+1}' for i in range(self.latent_size)],
            output_labels=[f'(signal) Control dimension {i+1}' for i in range(self.control_dim)],
        )

        self.register_method(
            "decode",
            in_channels=self.control_dim,
            in_ratio=self.ae_ratio,
            out_channels=self.latent_size,
            out_ratio=self.ae_ratio,
            input_labels=[f'(signal) Control dimension {i+1}' for i in range(self.control_dim)],
            output_labels=[f'(signal) Latent dimension {i+1}' for i in range(self.latent_size)],
        )

        self.register_method(
            "forward",
            in_channels=self.latent_size,
            in_ratio=self.ae_ratio,
            out_channels=self.latent_size,
            out_ratio=self.ae_ratio,
            input_labels=[f'(signal) Latent dimension {i+1}' for i in range(self.latent_size)],
            output_labels=[f'(signal) Latent dimension {i+1}' for i in range(self.latent_size)],
        )
    
    @torch.jit.export
    def encode(self, z):
        c, _ = self.encoder(z)
        return c

    @torch.jit.export
    def decode(self, c):
        z = self.decoder(c)
        return z
    
    @torch.jit.export
    def forward(self, z):
        return self.decode(self.encode(z))
    

@click.command()
@click.option("--ckpt", default="", help="path to checkpoint")
@click.option("--config", default="", help="path to config.gin")
@click.option("-n", "--name", default="model", help="name of the model")
@click.option("--ae_ratio", default=2048, help="AE ratio")
@click.option("--streaming", is_flag=True, help="activate causal cached conv")
def main(ckpt, config, name, ae_ratio, streaming):
    cc.use_cached_conv(streaming)

    # parse config
    gin.parse_config_file(config)
    
    # instantiate model
    pretrained = CoALa()

    # load checkpoints
    checkpoint = torch.load(ckpt, map_location="cpu")
    pretrained.load_state_dict(checkpoint["state_dict"], strict=False)
    pretrained.eval()

    for m in pretrained.modules():
        if hasattr(m, "weight_g"):
            nn.utils.remove_weight_norm(m)

    # scripted model
    scripted = ScriptedCoALa(pretrained, ae_ratio)

    max_length = 2**16 // ae_ratio
    z = torch.randn(1, pretrained.latent_size, max_length)
    print(z.shape)
    c = scripted.encode(z)
    print(c.shape)
    z_hat = scripted.decode(c)
    print(z_hat.shape)
    
    model_name = name
    if streaming:
        model_name += "_streaming"
    model_name += ".ts"
    output_path = os.path.join(os.path.dirname(config), model_name)
    scripted.export_to_ts(output_path)

    print(f"Done! Model exported to {output_path}")
    

if __name__ == '__main__':
    main()
