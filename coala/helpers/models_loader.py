import os
import gin
import torch
import pickle
from typing import Union
from music2latent import EncoderDecoder

from coala.models.model import CoALa

from coala.eval.platune import PLaTune


def load_model(
        ckpt_path: str,
        config_path: str,
        emb_model_path: str = 'music2latent',
        device: Union[str, torch.device] = 'cpu',
        version : str = 'v1',
    ):
    gin.parse_config_file(config_path)
    checkpoint = torch.load(ckpt_path, map_location=device)

    if version == 'v1':
        pretrained = CoALa()
    else:
        raise NotImplementedError(f'No model for version {version}')
    pretrained.load_state_dict(checkpoint["state_dict"], strict=False)
    pretrained.eval()
    pretrained = pretrained.to(device)

    if emb_model_path == "music2latent":
        emb_model = EncoderDecoder(device=device)
    else:
        emb_model = torch.jit.load(emb_model_path).to(device)

    return pretrained, emb_model


def load_platune(
        ckpt_path: str,
        config_path: str,
        emb_model_path: str,
        device: Union[str, torch.device] = 'cpu',
        quantized: bool = True,
    ):
    gin.parse_config_file(config_path)
    checkpoint = torch.load(ckpt_path, map_location=device)

    bins_values = []
    data_path = gin.query_parameter('audio_dataset.load_data.data_path')
    print(data_path)
    bins_filepath = os.path.join(data_path, "bins_values.pkl")
    if quantized:
        continuous_keys = gin.query_parameter('%CONTINUOUS_KEYS')
        
        with open(bins_filepath, "rb") as f:
            bins = pickle.load(f)

        for k, v in bins.items():
            if k in continuous_keys:
                bins_values.append(v)

    if len(bins_values) > 0:
        gin.bind_parameter(
            "platune.PLaTune.bins_values",
            bins_values
        )
    pretrained = PLaTune()

    pretrained.load_state_dict(checkpoint["state_dict"])
    pretrained.eval()
    pretrained = pretrained.to(device)

    if emb_model_path == "music2latent":
        emb_model = EncoderDecoder(device=device)
    else:
        emb_model = torch.jit.load(emb_model_path).to(device)

    return pretrained, emb_model, data_path