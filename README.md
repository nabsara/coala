# CoALa: A Plugin model for Controlling Audio Latents

This repository provides the code and resources for our paper accepted at [ISMIR 2026](https://ismir2026.ismir.net/). The [paper](https://ismir2026program.ismir.net/poster_315.html#paper), [poster](https://ismir2026program.ismir.net/poster_315.html#poster) and [video presentation](https://ismir2026program.ismir.net/poster_315.html#video) are availble on the conference website. Listen to audio examples on our [GitHub page](https://nabsara.github.io/coala/).

## Abstract 
In recent years, neural audio synthesis has advanced significantly, enabling high-quality real-time waveform generation on standard CPUs. However, defining personalized controls for live musical performance remains challenging, as existing methods typically require large labeled datasets and rely on heavy architectures that are generally impractical for real-time use. In this paper, we propose CoALa, a lightweight plugin model for Controlling Audio Latents in real-time, which enables users to add time-varying musical controls on pretrained neural audio synthesis models using only a limited set of labeled examples. Our method leverages a variational autoencoder with a contrastive learning regularization to disentangle control-relevant features from the pretrained latent space in designated dimensions. We rely on labeled frames as an augmentation strategy to generate multiple views of similar control variations for each unlabeled latent sequence, and apply the contrastive objective on these variation embeddings. By augmenting the pretrained latent space with temporal latent tokens, our method supports an arbitrary number of control variables, generalizes to highly compact latent representations, and can simultaneously disentangle multiple time-varying controls evolving at different temporal scales. Our model defines an invertible mapping between both latent and control trajectories and achieves high control accuracy while preserving synthesis quality for feature extraction, editing, and conditional synthesis tasks, using as little as 10\% labeled data. Its lightweight streamable architecture further enables fast CPU inference, making it well suited for real-time applications in Max/MSP environments.

## Install

Create a virtual environment with Python 3.12.9 and install the dependencies with:
```bash
(myenv)$ pip install -r requirements.txt
```

- To use GPU, install the torch version compatible with your cudnn version using:
```bash
(myenv)$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
(change the `index-url` if necessary)


## Data Processing


1. process audio files to lmdb dataset
```bash
(myenv)$ python scripts/process_dataset.py -i data/medley_solos/audios -o data/medley_solos/processed -s 10 -p medley_solos_mono_parser -m music2latent --gpu 0 -n 131072 --sr 44100 -b 32 --cut_silences --save_waveform
```

2. compute and save descriptors with lmdb audio examples
```bash
(myenv)$ python scripts/process_descriptors.py --data_path data/medley_solos/processed/ -o data/medley_solos/processed_desc
```

3. compute min-max, equal bin quantization, label class proportions
```bash
(myenv)$ python scripts/compute_min_max_descriptors.py --data_path data/medley_solos/processed_desc --n_bins 20
```

4. split dataset using lmdb keys
- split val / train
```bash
(myenv)$ python scripts/save_lmdb_keys.py --db_path data/medley_solos/processed_desc/ --n1 val --n2 train -r 0.1 --seed 42
```
- split label / no label

for 10% labeled data use:
```bash
(myenv)$ python scripts/save_lmdb_keys.py --db_path data/medley_solos/processed_desc/ --n1 label --n2 nolabel -r 0.1 --seed 42 -l lmdb_keys_s42_r90_train.pkl
```

for 50% labeled data use:
```bash
(myenv)$ python scripts/save_lmdb_keys.py --db_path data/medley_solos/processed_desc/ --n1 label --n2 nolabel -r 0.5 --seed 42 -l lmdb_keys_s42_r90_train.pkl
```

After completing the data preprocessing steps, you should have the following directory structure:
```
data/
└── medley_solos/
    ├── processed/
    │   ├── data.mdb
    │   └── lock.mdb
    └── processed_desc/
        ├── bins_values.pkl
        ├── data.mdb
        ├── lmdb_keys_s42_r10_label.pkl
        ├── lmdb_keys_s42_r10_val.pkl
        ├── lmdb_keys_s42_r50_label.pkl
        ├── lmdb_keys_s42_r50_nolabel.pkl
        ├── lmdb_keys_s42_r90_nolabel.pkl
        ├── lmdb_keys_s42_r90_train.pkl
        ├── lmdb_keys.pkl
        ├── lock.mdb
        ├── metadata_attributes.json
        └── skip_keys.pkl
``` 

# Training

We specify the training configurations in a `*.gin` file, for instance `coala_m2l_rms_centroid_bandwidth.gin` in this [folder](./coala/configs/).

You can update the following control attributes parameters:
```python
CDIM = 3
LATENT_DIM = 64  # Music2Latent pretrained codec
CONTROL_DIM = 67

ATTRIBUTES = ['rms', 'centroid', 'bandwidth']

N_CLASSES_PER_FACTOR = [20, 20, 20]
QUANTIZED = [True, True, True]
```

Then, launch the training script with:

```bash
(myenv)python scripts/train.py -d data/medley_solos/processed_desc -n coala_m2lms -c ismir/coala_m2l_rms_centroid_bandwidth -s exp --max_steps 250000 --val_every 10000 --ckpt_steps 50000 --gpu 1
```

# Export model to torchscript for real-time streaming in MaxMSP using nn~

CoALa can be trained on the latent space of a pretrained [RAVE](https://github.com/acids-ircam/RAVE) model and then use for real-time streaming control in MaxMSP using [nn~](https://github.com/acids-ircam/nn_tilde). Some pretrained RAVE models are publicly available [here](https://acids-ircam.github.io/rave_models_download).

Once training is complete, open the `config.gin` file and manually update the `stream` parameter from `False` to `True`:

```python
...
networks.CNNLatentEncoderWithExtraLatents.stream = True
...
```

Then, launch the export script:
```bash
(myenv)$ python scripts/export.py --ckpt exp/coala_rave/version_0/checkpoints/step=0300000.ckpt --config exp/coala_rave/config.gin -n coala_rave --ae_ratio 2048 --streaming
```

**FIX: You need to update the `export_cc_convs.py` file in the nn_tilde package:**
```python
...

class AlignBranches(nn.Module):

  ...

  # def forward(self, x):
  #     outs = []
  #     for branch, pad in zip(self.branches, self.paddings):
  #         delayed_x = pad(x)
  #         outs.append(branch(delayed_x))
  #     return outs

  def forward(self, x, cond):
      outs = []
      for branch, pad in zip(self.branches, self.paddings):
          delayed_x = pad(x)
          outs.append(branch(delayed_x, cond))
      return outs

class Branches(nn.Module):

  ...

  # def forward(self, x):
  #     outs = []
  #     for branch in self.branches:
  #         outs.append(branch(x))
  #     return outs

  def forward(self, *args):
      outs = []
      for branch in self.branches:
          outs.append(branch(*args))
      return outs

```

# Citation
```
@inproceedings{nabi2026coala,
  title={CoALa: A Plugin model for Controlling Audio Latents},
  author={Nabi, Sarah and Goswami, Nabarun and Esling, Philippe and Peeters, Geoffroy and Bevilacqua, Fr{\'e}d{\'e}ric and Harada, Tatsuya},
  booktitle={International Society for Music Information Retrieval, ISMIR 2026},
  year={2026}
}
```