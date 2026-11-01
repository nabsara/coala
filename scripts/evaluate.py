import os
import click
import gin
import torch
import json
import torch.nn.functional as F
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchaudio

from coala.helpers.models_loader import load_model, load_platune
from coala.datasets.audio_dataset import AudioDataset
from coala.datasets.audio_descriptors import compute_audio_descriptors


CKPT_STEPS = '0300000'
MODEL_NAME = 'coala_m2lms'
EXP_PATH = 'TO-CHANGE'
CKPT = f'{EXP_PATH}/{MODEL_NAME}/version_0/checkpoints/step={CKPT_STEPS}.ckpt'
CONFIG = '{EXP_PATH}/{MODEL_NAME}/config.gin'

EMB_MODEL = 'music2latent'
DEVICE = 'cuda'

SAMPLE_RATE = 44100
N_SIGNAL = 131072
AE_RATIO = 4096
Z_LENGTH = 31


LMBD_KEYS = 'lmdb_keys_s42_r10_val.pkl'
BATCH = 32
N_WORKERS = 8

CDIMS_SWAP = 1


def compute_y(w, sr, ae_ratio, z_length, descriptors, is_pow_2, device):
    w_np = w.detach().squeeze().cpu().numpy()
    features = compute_audio_descriptors(
        y=w_np,
        sr=sr,
        ae_ratio=ae_ratio,
        z_length=z_length,
        descriptors=descriptors,
        instrument_val=None,
        pm=None,
        device='cpu',
    )
    attr = []
    for k in descriptors:
        a = torch.from_numpy(features[k])
        attr.append(a)
    attr = torch.stack(attr)
    if not is_pow_2:
        attr = torch.nn.functional.pad(attr, (0, 1), mode='replicate')
    return attr.to(device)
    
def normalize(x, eps=1e-8):
    # x: (D, T)
    mean = x.mean(dim=-1, keepdim=True)   # per dim over time
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + eps)


def eval_feature_extraction(y, c, descriptors, normalize_inputs=True, eps=1e-8):
    """
    y, c: (B, D, T)
    descriptors: list of length D

    returns:
        dict: {descriptor: {mse, l1, corr}}
    """

    assert y.shape == c.shape
    B, D, T = y.shape
    assert len(descriptors) == D

    results = {}

    for d in range(D):
        name = descriptors[d]

        # extract one descriptor (B, T)
        y_d = y[:, d, :]
        c_d = c[:, d, :]

        if normalize_inputs:
            y_d = normalize(y_d, eps)
            c_d = normalize(c_d, eps)

        # reshape (B*T,)
        y_flat = y_d.reshape(-1)
        c_flat = c_d.reshape(-1)

        # MSE
        mse = F.mse_loss(y_flat, c_flat, reduction="mean")

        # L1
        l1 = F.l1_loss(y_flat, c_flat, reduction="mean")

        # Correlation (per sample, then average)
        y_centered = y_d - y_d.mean(dim=1, keepdim=True)
        c_centered = c_d - c_d.mean(dim=1, keepdim=True)

        numerator = (y_centered * c_centered).sum(dim=1)
        denominator = (
            torch.norm(y_centered, dim=1) * torch.norm(c_centered, dim=1) + eps
        )

        corr = (numerator / denominator).mean()

        results[name] = {
            "mse": mse.item(),
            "l1": l1.item(),
            "corr": corr.item(),
        }

    return results


def compute_metrics_dt(y, y_gen, normalize_inputs=True, eps=1e-8):
    """
    y, y_gen: (D, T)

    reduction:
        - MSE/L1: mean over T, then mean over D
        - Corr: computed over time per descriptor, then mean over D
    """

    assert y.shape == y_gen.shape
    D, T = y.shape

    if normalize_inputs:
        y = normalize(y, eps)
        y_gen = normalize(y_gen, eps)

    # MSE (per dim over time)
    mse = F.mse_loss(y, y_gen, reduction="none").mean(dim=1).mean()

    # L1
    l1 = F.l1_loss(y, y_gen, reduction="none").mean(dim=1).mean()

    # Correlation (over time, per descriptor)
    y_centered = y - y.mean(dim=1, keepdim=True)
    ygen_centered = y_gen - y_gen.mean(dim=1, keepdim=True)

    numerator = (y_centered * ygen_centered).sum(dim=1)  # over T
    denominator = (
        torch.norm(y_centered, dim=1) * torch.norm(ygen_centered, dim=1) + eps
    )

    corr_per_dim = numerator / denominator
    corr = corr_per_dim.mean()

    return {
        "mse": mse.item(),
        "l1": l1.item(),
        "corr": corr.item(),
    }


def swap_control_attributes(c, y, cdim=3, mode="one"):
    """
    c: (B, D, T)
    y: (B, D, T)  (or same structure for labels)

    returns:
        c_swap, y_swap
    """

    B, D, T = c.shape
    device = c.device

    c_swap = c.clone()
    y_swap = y.clone()

    if mode == "all":
        # one permutation per control dim
        perms = torch.stack(
            [torch.randperm(B, device=device) for _ in range(cdim)],
            dim=1
        )  # (B, cdim)

        dim_idx = torch.arange(cdim, device=device)

        # apply same swap to c and y
        c_swap[:, :cdim, :] = c[perms, dim_idx, :]
        y_swap[:, :cdim, :] = y[perms, dim_idx, :]

    elif mode == "one":
        perm = torch.randperm(B, device=device)
        dims = torch.randint(0, cdim, (B,), device=device)

        b_idx = torch.arange(B, device=device)

        c_swap[b_idx, dims, :] = c[perm[b_idx], dims, :]
        y_swap[b_idx, dims, :] = y[perm[b_idx], dims, :]

    else:
        raise ValueError("mode must be 'one' or 'all'")

    return c_swap, y_swap


def create_save_paths(save_path, name, suffix_list):
    paths = {
        suffix: os.path.join(save_path, f'{name}_{suffix}')
        for suffix in suffix_list
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


@click.command()
@click.option('--ckpt', default=CKPT)
@click.option('--config', default=CONFIG)
@click.option('--emb_model', default=EMB_MODEL)
@click.option('--device', default=DEVICE)
@click.option('--lmdb_keys_filename', default=LMBD_KEYS)
@click.option('--batch_size', default=BATCH)
@click.option('--n_workers', default=N_WORKERS)
@click.option('--save_gt', is_flag=True)
@click.option('--save_emb', is_flag=True)
@click.option('--cdims_swap', default=CDIMS_SWAP, help='int that represent the nb control dimensions index to swap')
@click.option('--sr', default=SAMPLE_RATE)
@click.option('--n_signal', default=N_SIGNAL, help='signal audio length')
@click.option('--z_length', default=Z_LENGTH)
@click.option('--ae_ratio', default=AE_RATIO)
@click.option('--eval_synth', is_flag=True)
@click.option('--eval_platune', is_flag=True)
@click.option('--ckpt_steps', default=CKPT_STEPS)
def eval(ckpt, config, emb_model, device, lmdb_keys_filename, batch_size, n_workers, save_gt, save_emb, cdims_swap, sr, n_signal, z_length, ae_ratio, eval_synth, eval_platune, ckpt_steps):
    torch.set_grad_enabled(False)

    # create save paths directories
    save_path = os.path.dirname(config)
    save_path = os.path.join(save_path, f'evaluation_{ckpt_steps}')
    print(save_path)
    os.makedirs(save_path, exist_ok=True)
    
    # cdims_swap = list(cdims_swap)
    print("cdims swap: ", cdims_swap)

    # load model
    if eval_platune:
        model, codec, data_path = load_platune(
            ckpt_path=ckpt,
            config_path=config,
            emb_model_path=emb_model,
            device=device,
            quantized=True,
        )
    else:
        model, codec = load_model(ckpt, config, emb_model, device, version='v1')

        # load data
        data_path = gin.query_parameter('val_data/audio_dataset.load_data.data_path')
    print(data_path)

    if lmdb_keys_filename is not None:
        lmdb_keys_path = os.path.join(data_path, lmdb_keys_filename)
        if os.path.isfile(lmdb_keys_path):
            with open(lmdb_keys_path, "rb") as f:
                lmdb_keys = pickle.load(f)
    else:
        lmdb_keys = []

    if eval_platune:
        descriptors = gin.query_parameter('%CONTINUOUS_KEYS')
    else:
        descriptors = gin.query_parameter('%ATTRIBUTES')
    print("descriptors: ", descriptors)
    data = AudioDataset(
        path=data_path,
        attr_keys=descriptors,
        other_keys=['metadata', 'waveform'],
        lmdb_keys=lmdb_keys,
        crop_length=None,
    )

    print(f'nb examples ({lmdb_keys_filename}): {len(data)}')

    data_loader = DataLoader(data, batch_size, shuffle=False, num_workers=n_workers, drop_last=False, persistent_workers=True if n_workers > 0 else False)

    swap_suffix = []
    if save_gt:
        swap_suffix.append('gt')
    if save_emb:
        swap_suffix.append('emb')
    if len(descriptors) > 1:
        swap_suffix.append('edit_swap_all')
    if eval_synth:
        swap_suffix.append('synth')
    audio_paths = create_save_paths(
        save_path, 
        name='audio', 
        suffix_list=['rec', 'edit_swap'] + swap_suffix
    )

    n = 0
    n_batch = 0

    metadata = {}

    loss_rec_z = 0.

    loss_extract = {d: {'mse': 0., 'l1': 0., 'corr': 0.} for d in descriptors}

    loss_y_rec = {'mse': 0., 'l1': 0., 'corr': 0.}
    if eval_synth:
        loss_y_synth = {'mse': 0., 'l1': 0., 'corr': 0.}
    loss_y_edit_swap = {'mse': 0., 'l1': 0., 'corr': 0.}
    if len(descriptors) > 1:
        loss_y_edit_swap_all = {'mse': 0., 'l1': 0., 'corr': 0.}
    
    is_pow_2 = (z_length > 0 and (z_length & (z_length - 1)) == 0)

    for batch in tqdm(data_loader):
        # DEBUG!
        # if n > 1:
        #     break
        n_batch += 1
        with torch.no_grad():
            z, y, _, w_gt = batch
            z, y, w_gt = z.to(device), y.to(device), w_gt.unsqueeze(1).to(device)

            if save_emb:
                # forward codec
                w_emb = codec.decode(z)[...,:n_signal]
            
            # REC 
            if eval_platune:
                c = model.z_to_cs(z, nb_steps=60)
                z_rec = model.cs_to_z(c, nb_steps=60)
            else:
                c, _ = model.control_extractor(z)
                z_rec = model.latent_projector(c)
            loss_rec_z_batch = torch.nn.functional.mse_loss(z, z_rec, reduction="none").sum(1).mean().item()
            loss_rec_z += loss_rec_z_batch

            w_rec = codec.decode(z_rec)[...,:n_signal]

            loss_extract_curr = eval_feature_extraction(y, c[:,:cdims_swap,:], descriptors)
            for d in descriptors:
                for k in loss_extract[d]:
                    loss_extract[d][k] += loss_extract_curr[d][k]
            
            
            # SYNTH
            if eval_synth:
                c_randn = torch.randn_like(c)
                c_randn[:,:cdims_swap,:] = c[:,:cdims_swap,:].clone()
                if eval_platune:
                    z_synth = model.cs_to_z(c_randn, nb_steps=60)
                else:
                    z_synth = model.latent_projector(c_randn)
                w_synth = codec.decode(z_synth)[...,:n_signal]


            # EDIT SWAP 1
            c_swap, y_swap = swap_control_attributes(c, y, cdim=cdims_swap, mode='one')
            if eval_platune:
                z_swap = model.cs_to_z(c_swap, nb_steps=60)
            else:
                z_swap = model.latent_projector(c_swap)
            w_swap = codec.decode(z_swap)[...,:n_signal]


            # EDIT SWAP ALL
            if len(descriptors) > 1:
                c_swap_all, y_swap_all = swap_control_attributes(c, y, cdim=cdims_swap, mode='all')
                if eval_platune:
                    z_swap_all = model.cs_to_z(c_swap_all, nb_steps=60)
                else:
                    z_swap_all = model.latent_projector(c_swap_all)
                w_swap_all = codec.decode(z_swap_all)[...,:n_signal]

             
        # SAVE
        for i in range(z.shape[0]):
            if save_gt:
                # 1. original waveform w_gt
                torchaudio.save(os.path.join(audio_paths['gt'], f"{n:05d}_gt.wav"), w_gt[i].detach().cpu(), sr)

            if save_emb:
                # 2. m2l(w_gt) --> save w_emb
                torchaudio.save(os.path.join(audio_paths['emb'], f"{n:05d}_emb.wav"), w_emb[i].unsqueeze(0).detach().cpu(), sr)
            
            # w_rec
            torchaudio.save(os.path.join(audio_paths['rec'], f"{n:05d}_rec.wav"), w_rec[i].unsqueeze(0).detach().cpu(), sr)

            y_rec_i = compute_y(w_rec[i], sr, ae_ratio, z_length, descriptors, is_pow_2, device)
            loss_y_rec_i = compute_metrics_dt(y[i], y_rec_i, normalize_inputs=True)

            for k in loss_y_rec:
                loss_y_rec[k] += loss_y_rec_i[k]

            # w_synth
            if eval_synth:
                torchaudio.save(os.path.join(audio_paths['synth'], f"{n:05d}_synth.wav"), w_synth[i].unsqueeze(0).detach().cpu(), sr)

                y_synth_i = compute_y(w_synth[i], sr, ae_ratio, z_length, descriptors, is_pow_2, device)
                loss_y_synth_i = compute_metrics_dt(y[i], y_synth_i, normalize_inputs=True)

                for k in loss_y_synth:
                    loss_y_synth[k] += loss_y_synth_i[k]



            # w_swap
            torchaudio.save(os.path.join(audio_paths['edit_swap'], f"{n:05d}_edit_swap.wav"), w_swap[i].unsqueeze(0).detach().cpu(), sr)

            y_swap_i = compute_y(w_swap[i], sr, ae_ratio, z_length, descriptors, is_pow_2, device)
            loss_y_swap_i = compute_metrics_dt(y_swap[i], y_swap_i, normalize_inputs=True)

            for k in loss_y_edit_swap:
                loss_y_edit_swap[k] += loss_y_swap_i[k]


            if len(descriptors) > 1:
                # w_swap_all
                torchaudio.save(os.path.join(audio_paths['edit_swap_all'], f"{n:05d}_edit_swap_all.wav"), w_swap_all[i].unsqueeze(0).detach().cpu(), sr)

                y_swap_all_i = compute_y(w_swap_all[i], sr, ae_ratio, z_length, descriptors, is_pow_2, device)
                loss_y_swap_all_i = compute_metrics_dt(y_swap_all[i], y_swap_all_i, normalize_inputs=True)

                for k in loss_y_edit_swap_all:
                    loss_y_edit_swap_all[k] += loss_y_swap_all_i[k]
            
            n += 1

    metadata["eval_data_path"] = save_path
    metadata["n_samples"] = n 
    metadata["inputs"] = {
        "ckpt_path": ckpt,
        "config_path": config,
        "sr": sr,
        "n_signal": n_signal,
        "emb_model": emb_model,
        "data_path": data_path,
        "descriptors": descriptors,
        "cdims_swap": cdims_swap
    }
    metadata["metrics"] = {
        "loss_rec_z": round(loss_rec_z / n_batch, 4)
    }
    
    metadata["metrics"]["extract"] = {}

    for d in descriptors:
        metadata["metrics"]["extract"][d] = {
            k: round(loss_extract[d][k] / n_batch, 4)
            for k in loss_extract[d]
        }

    metadata["metrics"]['rec'] = {k: round(v / n, 4) for k, v in loss_y_rec.items()}
    if eval_synth:
        metadata["metrics"]['synth'] = {k: round(v / n, 4) for k, v in loss_y_synth.items()}
    metadata["metrics"]['edit_swap'] = {k: round(v / n, 4) for k, v in loss_y_edit_swap.items()}
    if len(descriptors) > 1:
        metadata["metrics"]['edit_swap_all'] = {k: round(v / n, 4) for k, v in loss_y_edit_swap_all.items()}
    
    print(metadata["metrics"])

    with open(os.path.join(save_path, 'metadata_results.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

    
        

if __name__ == '__main__':
    eval()
