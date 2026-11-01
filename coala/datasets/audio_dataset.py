import os
import gin
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
from typing import List, Union
from tqdm import tqdm

from .data_example import DataExample
from .lmdb_dataset import LmdbDataset


def force_onsets_every_n(mask, n_max_frames):
        batch_size, seq_length = mask.shape
        out = mask.clone()

        for b in range(batch_size):
            count = 0
            for t in range(seq_length):
                if out[b, t]:
                    count = 0
                else:
                    count += 1
                    if count >= n_max_frames:
                        out[b, t] = True
                        count = 0

        return out


class AudioDataset(LmdbDataset):

    def __init__(self, path, attr_keys, other_keys = [], lmdb_keys = [], crop_length=None):
        
        keys = ['z']
        keys += attr_keys
        keys += other_keys

        super().__init__(path, keys, lmdb_keys=lmdb_keys)
        # super().__init__(path, "all", lmdb_keys=lmdb_keys)
        
        self.attr_keys = attr_keys

        self.crop_length = crop_length
        
        self.attr_cache = False
        self.seg_cache = False

    def _convert_class_values_to_indices(self, a, class_values):
        new_a = []
        for i in range(a.shape[0]):
            if len(class_values[i]) > 0:
                classes_i = torch.tensor(class_values[i])
                a_i = a[i, :]
                a_ids = torch.searchsorted(classes_i.contiguous(), a_i.contiguous())
            else:
                a_ids = a[i, :]
            new_a.append(a_ids)
        new_a = torch.stack(new_a)
        return new_a
    
    def process_attributes(self, class_values, quantized, bins_values):
        print('prepocess raw attributes to assign int class labels')
        self.class_values = class_values
        self.quantized = quantized
        self.bins_values = [torch.tensor(v) for v in bins_values]
        self.attr_processed = []
        for i in tqdm(range(len(self))):
            _, y = self.__getitem__(i)
            attr = y.clone()
            if class_values is not None:
                attr = self._convert_class_values_to_indices(attr, class_values)

            # assign class value to quantized attributes
            for k in range(y.shape[0]):
                if quantized[k]:
                    data = attr[k, :]
                    classes = torch.bucketize(data, self.bins_values[k])
                    attr[k, :] = classes.reshape(-1, attr.shape[-1])
                    
            attr = attr.to(torch.int32)
            self.attr_processed.append(attr)
        self.attr_cache = True

    def precompute_segments(self, n_max_frames):
        print('precompute segments to efficiently aggregate consecutive frames during training')
        assert self.attr_cache
        self.n_max_frames = n_max_frames
        self.seg_ids = []
        for i in tqdm(range(len(self))):
            a = self.attr_processed[i]
            seg_ids_i = []
            for k in range(a.shape[0]):
                a_k = a[k]  # (T,)

                # detect segment boundaries
                change = torch.ones(a.shape[1], dtype=torch.bool)
                change[1:] = a_k[1:] != a_k[:-1]

                if self.n_max_frames[k] is not None:
                    change = force_onsets_every_n(
                        change.unsqueeze(0), self.n_max_frames[k]
                    )[0]
                # segment ids
                s_ids = change.cumsum(0) - 1  # (T,)
                seg_ids_i.append(s_ids)
            seg_ids_i = torch.stack(seg_ids_i).to(torch.int64)
            self.seg_ids.append(seg_ids_i)
        self.seg_cache = True

    def __getitem__(self, idx):
        if self.cache:
            return self.data[idx]

        with self.env.begin() as txn:
            data_ex = DataExample(txn.get(self.keys[idx]))

        z = torch.from_numpy(data_ex.get("z"))
        z_length = z.shape[-1]

        is_pow_2 = (z_length > 0 and (z_length & (z_length - 1)) == 0)
        if not is_pow_2:
            z = torch.nn.functional.pad(z, (0, 1), mode='replicate')
            z_length = z.shape[-1]

        if not self.attr_cache:
            attr = []

            for key in self.buffer_keys:
                if key in ['z', 'metadata', 'waveform', 'midi']:
                    continue
                
                try:
                    a = torch.from_numpy(data_ex.get(key))
                    attr.append(a)

                except:
                    raise ValueError(f'Do not know how to handle key={key}')

            if len(attr) > 0:
                attr = torch.stack(attr)
                if not is_pow_2:
                    attr = torch.nn.functional.pad(attr, (0, 1), mode='replicate')
            else:
                attr = torch.tensor(attr)
                
            if self.crop_length is not None:
                id_crop = np.random.randint(0, z_length - self.crop_length)
                z = z[:, id_crop:id_crop + self.crop_length]
                if len(attr) > 0:
                    attr = attr[:,id_crop:id_crop + self.crop_length]
            
            others = []
            
            if 'metadata' in self.buffer_keys:
                metadata = data_ex.get_metadata()
                others.append(metadata)
            
            if 'waveform' in self.buffer_keys:
                w = data_ex.get('waveform').astype(np.float32) / (2**15 - 1)
                w = torch.from_numpy(w)
                others.append(w)
            
            if 'midi' in self.buffer_keys:
                midi = data_ex.get('midi')
                others.append(midi)
            
            if len(others) > 0:
                return z, attr, *others
            else:
                return z, attr
        
        else:
            if not self.seg_cache:
                return z, self.attr_processed[idx]
            else:
                return z, self.attr_processed[idx], self.seg_ids[idx]


@gin.configurable
def load_data(
        data_path: str, 
        attr_keys : List[str] = ['centroid'], 
        other_keys: List[str] = ['metadata'], 
        lmdb_keys_filename : str = None,
        crop_length : int = None,
        batch_size: int = 8,
        shuffle : bool = True, 
        n_workers: int = 0, 
        drop_last :bool = False,
        use_cache : bool = False,
        class_values : List[int] = None, 
        quantized : List[bool] = None, 
        bins_values : List[Union[List[None], torch.Tensor]] = [],
        n_max_frames : List[int] = None,
    ):

    if lmdb_keys_filename is not None:
        lmdb_keys_path = os.path.join(data_path, lmdb_keys_filename)
        if os.path.isfile(lmdb_keys_path):
            with open(lmdb_keys_path, "rb") as f:
                lmdb_keys = pickle.load(f)
    else:
        lmdb_keys = []

    dataset = AudioDataset(
        path=data_path,
        attr_keys=attr_keys,
        other_keys=other_keys,
        lmdb_keys=lmdb_keys,
        crop_length=crop_length,
    )

    print(f'nb examples ({lmdb_keys_filename}): {len(dataset)}')

    if use_cache:
        dataset.process_attributes(class_values=class_values, quantized=quantized, bins_values=bins_values)
        dataset.precompute_segments(n_max_frames=n_max_frames)

    data_loader = DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=n_workers, drop_last=drop_last, persistent_workers=True if n_workers > 0 else False)

    return data_loader
