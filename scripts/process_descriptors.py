import os
import lmdb
import click
import pickle
import numpy as np
from tqdm import tqdm

from coala.datasets.lmdb_dataset import LmdbDataset
from coala.datasets.data_example import DataExample
from coala.datasets.audio_descriptors import compute_audio_descriptors


SR = 44100
Z_LENGTH = 31
AE_RATIO = 4096

DESCRIPTORS = [
    "rms",
    "loudness",
    "centroid", 
    "bandwidth", 
    # "centroid-filt", 
    # "bandwidth-filt", 
    "booming",
    "sharpness",
]

KEYS = [
    # 'midi',
]

DEVICE = "cuda"

# INSTRUMENTS = []
# MedleySolosDB
INSTRUMENTS = ['female singer', 'violin', 'flute', 'clarinet', 'tenor saxophone', 'trumpet']


@click.command()
@click.option('--data_path', default="", help='path to input dataset')
@click.option('-o', '--output_path', default="", help='path to output dataset')
@click.option('-s', '--db_size', default=10, help='Max Size of lmdb database')
def main(data_path, output_path, db_size):
    dataset = LmdbDataset(  
        path=data_path,
        keys=['waveform', 'metadata', 'z'] + KEYS
    )

    # initialize lmdb database
    os.makedirs(output_path, exist_ok=True)
    env = lmdb.open(
        output_path,
        map_size=db_size * 1024**3,
        map_async=True,
        writemap=True,
        readahead=False,
    )

    cur_index = 0
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        metadata = data['metadata']

        # create instance of our lmdb database
        key = f"{cur_index:08d}"
        data_ex = DataExample()
        
        # metadata
        metadata['new_key'] = key
        data_ex.put_metadata(metadata)

        # waveform
        w = data['waveform']
        data_ex.put_array("waveform", w, dtype=np.int16)
        
        # latent
        z = data['z']
        data_ex.put_array("z", z, dtype=np.float32)
        
        midi_data = None
        if "midi" in KEYS:
            midi_data = data["midi"]
            data_ex.put_buffer(key="midi", b=pickle.dumps(midi_data), shape=None)
        
        inst_class = None
        if "instrument" in metadata.keys():
            inst_class = INSTRUMENTS.index(metadata["instrument"])
            
        y = w.astype(np.float32) / (2**15 - 1)
        features = compute_audio_descriptors(
            y=y,
            sr=SR,
            ae_ratio=AE_RATIO,
            z_length=Z_LENGTH,
            descriptors=DESCRIPTORS,
            instrument_val=inst_class,
            pm=midi_data,
            device=DEVICE,
        )

        for k, v in features.items():
            if v.dtype == 'float32':
                t = np.float32
            elif v.dtype == 'int32':
                t = np.int32
            else:
                raise ValueError(f'Do not know how to store dtype={v.dtype} for descriptor={k}')
            data_ex.put_array(k, v, dtype=t)

        # save DataExample instance to lmdb database
        with env.begin(write=True) as txn:
            txn.put(key.encode(), bytes(data_ex))
        cur_index += 1
        
    print('nb_examples: ', cur_index)
    env.close()


if __name__ == '__main__':
    main()
