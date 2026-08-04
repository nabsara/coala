# CoALa: A Plugin model for Controlling Audio Latents

This repository is linked to our paper submission to the [ISMIR 26](https://ismir2026.ismir.net/) Conference. 

## Abstract 
In recent years, neural audio synthesis has advanced significantly, enabling high-quality real-time waveform generation on standard CPUs. However, defining personalized controls for live musical performance remains challenging, as existing methods typically require large labeled datasets and rely on heavy architectures that are generally impractical for real-time use. In this paper, we propose CoALa, a lightweight plugin model for Controlling Audio Latents in real-time, which enables users to add time-varying musical controls on pretrained neural audio synthesis models using only a limited set of labeled examples. Our method leverages a variational autoencoder with a contrastive learning regularization to disentangle control-relevant features from the pretrained latent space in designated dimensions. We rely on labeled frames as an augmentation strategy to generate multiple views of similar control variations for each unlabeled latent sequence, and apply the contrastive objective on these variation embeddings. By augmenting the pretrained latent space with temporal latent tokens, our method supports an arbitrary number of control variables, generalizes to highly compact latent representations, and can simultaneously disentangle multiple time-varying controls evolving at different temporal scales. Our model defines an invertible mapping between both latent and control trajectories and achieves high control accuracy while preserving synthesis quality for feature extraction, editing, and conditional synthesis tasks, using as little as 10\% labeled data. Its lightweight streamable architecture further enables fast CPU inference, making it well suited for real-time applications in Max/MSP environments.

## Install

Create a virtual environment with Python 3.12.9 and install the dependencies with:
```bash
(myenv)$ pip install -r requirements.txt
```

- To use GPU, install the torch version compatible with your cudnn version using:
```bash
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
(change the `index-url` if necessary)


## Data Processing


1. process audio files to lmdb dataset
```bash
python scripts/process_dataset.py -i data/medley_solos/audios -o data/medley_solos/processed -s 10 -p medley_solos_mono_parser -m music2latent --gpu 0 -n 131072 --sr 44100 -b 32 --cut_silences --save_waveform --use_basic_pitch
```

2. compute and save descriptors with lmdb audio examples
```bash
python scripts/process_descriptors.py --data_path data/medley_solos/processed/ -o data/medley_solos/processed_desc
```

3. compute min-max, equal bin quantization, label class proportions
```bash
python scripts/compute_min_max_descriptors.py --data_path
```

4. split dataset using lmdb keys
- split val / train
```bash
python scripts/save_lmdb_keys.py --db_path data/medley_solos/processed_desc/ --n1 val --n2 train -r 0.1 --seed 42
```
- split label / no label
```bash
python scripts/save_lmdb_keys.py --db_path data/medley_solos/processed_desc/ --n1 label --n2 nolabel -r 0.5 --seed 42 -l lmdb_keys_s42_r90_train.pkl
````

# Training

Example:

```bash
(myenv)$ export PYTHONPATH=$PYTHONPATH:$PWD
```

```bash
(myenv)$ python scripts/train.py 
    --db_path /<path-to>/data/processed
    -n model_name 
    -c template 
    -s /<path-to>/exp/ 
    --max_steps 100000 
    --val_every 1000 
    --ckpt_steps 10000 
    --gpu 1
```

[TO BE COMPLETED]


# Citation
```
@inproceedings{nabi2026coala,
  title={CoALa: A Plugin model for Controlling Audio Latents},
  author={Nabi, Sarah and Goswami, Nabarun and Esling, Philippe and Peeters, Geoffroy and Bevilacqua, Fr{\'e}d{\'e}ric and Harada, Tatsuya},
  booktitle={International Society for Music Information Retrieval, ISMIR 2026},
  year={2026}
}
```