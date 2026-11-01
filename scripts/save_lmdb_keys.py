import os
import click
import torch
import pickle
from tqdm import tqdm

from coala.datasets.lmdb_dataset import LmdbDataset


# MedleySolosDB
INSTRUMENTS = ['female singer', 'violin', 'flute', 'clarinet', 'tenor saxophone', 'trumpet']
# INSTRUMENTS = []


def split_idx(labels, ratio, seed : int = 42):
    g = torch.Generator()
    g.manual_seed(seed)

    idx1, idx2 = [], []

    for cls in torch.unique(labels, sorted=True):
        cls_idx = torch.where(labels == cls)[0]

        perm = cls_idx[torch.randperm(
            len(cls_idx), generator=g
        )]

        n1 = int(len(perm) * ratio)

        idx1.append(perm[:n1])
        idx2.append(perm[n1:])

    idx1 = torch.cat(idx1)
    idx2 = torch.cat(idx2)

    return idx1, idx2


@click.command()
@click.option('-d', '--db_path', default="", help='lmdb dataset path to split')
@click.option('--n1',default="val", help='name file saved for set 1')
@click.option('--n2',default="train", help='name file saved for set 2')
@click.option('-l', '--lmdb_keys_file', default=None, help='filepath to pickle with lmdb keys list')
@click.option('-r', '--ratio', default=0.5, help='split ratio between 0 and 1')
@click.option('-s', '--seed', default=42, help='seed torch generator in split')
def save_lmdb_keys_train_val_sets(db_path, n1, n2, lmdb_keys_file, ratio, seed):

    if lmdb_keys_file is not None and os.path.isfile(os.path.join(db_path, lmdb_keys_file)):
        with open(os.path.join(db_path, lmdb_keys_file), "rb") as f:
            lmdb_keys = pickle.load(f)
    else:
        lmdb_keys = []
        
    dataset = LmdbDataset(  
        path=db_path,
        keys=['metadata'],
        lmdb_keys=lmdb_keys
    )
        
    idx_inst_map = []
    if len(INSTRUMENTS) > 0:
        for i in tqdm(range(len(dataset))):
            d = dataset[i]
            inst = d["metadata"]["instrument"]
            inst_class = INSTRUMENTS.index(inst)
            idx_inst_map.append(inst_class)
        idx_inst_map = torch.tensor(idx_inst_map, dtype=torch.int32)

        idx1, idx2 = split_idx(idx_inst_map, ratio, seed)
    else:
        g = torch.Generator()
        g.manual_seed(seed)

        n_examples = len(dataset)
        perm = torch.randperm(n_examples, generator=g)
        n_split = int(n_examples * ratio)

        idx1 = perm[:n_split]
        idx2 = perm[n_split:]

    k  = 'key'
    if 'new_key' in dataset[0]["metadata"].keys():
        k = 'new_key'

    lmdb_keys1 = []
    lmdb_keys2 = []
    
    for i in tqdm(range(len(dataset))):
        d = dataset[i]
        lmdb_k = d["metadata"][k].encode("utf-8") 

        if i in idx1:
            lmdb_keys1.append(lmdb_k)
        elif i in idx2:
            lmdb_keys2.append(lmdb_k)

    print(f'nb examples set 1 - {n1}: {len(lmdb_keys1)}')
    print(f'nb examples set 2 - {n2}: {len(lmdb_keys2)}')

    with open(os.path.join(db_path, f"lmdb_keys_s{seed}_r{int(ratio*100)}_{n1}.pkl"), "wb") as f:
        pickle.dump(lmdb_keys1, f)
    
    with open(os.path.join(db_path, f"lmdb_keys_s{seed}_r{100 - int(ratio*100)}_{n2}.pkl"), "wb") as f:
        pickle.dump(lmdb_keys2, f)


if __name__ == '__main__':
    save_lmdb_keys_train_val_sets()
