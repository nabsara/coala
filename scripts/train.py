import gin
import os
import click
import pickle
import torch
import pytorch_lightning as pl
from pathlib import Path

from coala.models.model import CoALa, MyDataModule


def search_for_run(run_path, mode="last"):
    if run_path is None: return None
    if ".ckpt" in run_path: return run_path
    ckpts = map(str, Path(run_path).rglob("*.ckpt"))
    ckpts = filter(lambda e: mode in os.path.basename(str(e)), ckpts)
    ckpts = sorted(ckpts)
    if len(ckpts): return ckpts[-1]
    else: return None


@click.command()
@click.option('-d', '--db_path', default="", help='dataset path without labels (unsup)')
@click.option('--ds_path', default=None, help='dataset path with labels (sup)')
@click.option('-n', '--name', default="", help='Name of the run')
@click.option('-c', '--config', default="cvae_template", help='Name of the gin configuration file to use')
@click.option('-s', '--save_path', default="", help='path to save models checkpoints')
@click.option('--max_steps', default=300_000, help='Maximum number of training steps')
@click.option('--val_every', default=10_000, help='Checkpoint model every n steps')
@click.option('--gpu', default=-1, help='GPU to use')
@click.option('--ckpt', default=None, help='Path to previous checkpoint of the run')
@click.option('--ckpt_steps', default=10_000, help='save checkpoints every n steps')
@click.option('--version', default='v1', help='model version')
def main(
        db_path, 
        ds_path,
        name, 
        config, 
        save_path, 
        max_steps, 
        val_every,     
        gpu, 
        ckpt,
        ckpt_steps,
        version
    ):

    # load config
    config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "coala/configs", f"{config}.gin")
    print('loading config file : ', config_file)
    gin.parse_config_files_and_bindings([config_file], []) 
    
    
    # load data
    if ds_path is None:
        ds_path = db_path
    
    with gin.unlock_config():
        gin.bind_parameter("unlabel_data/audio_dataset.load_data.data_path", db_path)
        gin.bind_parameter("label_data/audio_dataset.load_data.data_path", ds_path)
        gin.bind_parameter("val_data/audio_dataset.load_data.data_path", ds_path)

    os.makedirs(os.path.join(save_path, name), exist_ok=True)

    # instantiate model
    bins_values = []
    bins_filepath = os.path.join(ds_path, "bins_values.pkl")
    if os.path.isfile(bins_filepath):
        with open(bins_filepath, "rb") as f:
            bins = pickle.load(f)
            attr = gin.query_parameter('%ATTRIBUTES')
            for a in attr:
                if a in bins.keys():
                    bins_values.append(torch.tensor(bins[a]).tolist())
                else:
                    bins_values.append([])

    with gin.unlock_config():
        if version == 'v1':
            gin.bind_parameter("model.CoALa.bins_values", bins_values)
        gin.bind_parameter("audio_dataset.load_data.bins_values", bins_values)
    

    # instantiate model
    dm = MyDataModule()
    if version == 'v1':
        model = CoALa()
    else:
        raise NotImplementedError(f'No model for version {version}')
    
    # model checkpoints
    callbacks_ckpt = []
    last_checkpoint = pl.callbacks.ModelCheckpoint(filename="last")
    callbacks_ckpt.append(last_checkpoint)

    checkpoint_n_steps = pl.callbacks.ModelCheckpoint(
        filename="{step:07}",
        save_top_k=-1,                     
        every_n_train_steps=int(ckpt_steps),          
        save_weights_only=False           
    )
    callbacks_ckpt.append(checkpoint_n_steps)

    val_check = {}
    n_batch = max(len(dm.train_loader_1), len(dm.train_loader_2))
    if n_batch >= val_every:
        val_check["val_check_interval"] = val_every
    else:
        nepoch = val_every // n_batch
        val_check["check_val_every_n_epoch"] = nepoch

    # select GPU
    accelerator =  "cuda" if torch.cuda.is_available() and gpu >= 0 else None
    if accelerator == "cuda":
        device = gpu
    print(f'device - selected gpu: {accelerator}:{device}')


    # instantiate trainer
    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(save_path, name=name),
        accelerator=accelerator,
        devices=device,
        callbacks=callbacks_ckpt,
        max_epochs=100000,
        max_steps=max_steps,
        profiler="simple",
        enable_progress_bar=True,
        # log_every_n_steps=10,
        **val_check,
    )

    run = search_for_run(ckpt)
    if run is not None:
        step = torch.load(run, map_location='cpu')["global_step"]
        print("Restarting from step : ", step)
        trainer.fit_loop.epoch_loop._batches_that_stepped = step

    with open(os.path.join(os.path.join(save_path, name), "config.gin"), "w") as config_out:
        config_out.write(gin.operative_config_str())

    # train model
    trainer.fit(model, ckpt_path=run, datamodule=dm)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    main()
