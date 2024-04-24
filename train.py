import argparse
import os
from time import strftime

import pytorch_lightning as pl
# from torchvision.transforms import ToTensor, Normalize

from codebase.datasets.unpaired import UnpairedDataset
from codebase.models.generative import ColorMapGAN, CycleGAN
from codebase.utils.transforms import ToTensor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
# from torchvision import transforms


def main():
    source_dir = PARAMS.source_dir
    target_dir = PARAMS.target_dir
    results_dir = PARAMS.results_dir
    epochs = PARAMS.epochs
    batch_size = PARAMS.batch_size
    lr_gen = PARAMS.lr_gen
    lr_dis = PARAMS.lr_dis
    model = PARAMS.model
    log_freq = PARAMS.log_freq
    enable_progress_bar = PARAMS.enable_progress_bar

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results_dir_root = os.path.dirname(results_dir.rstrip('/'))
    results_dir_name = os.path.basename(results_dir.rstrip('/'))

    exp_name = f"{model}-{strftime('%y%m%d')}-{strftime('%H%M%S')}"

    train_dataset = UnpairedDataset(
        source_dir=source_dir,
        target_dir=target_dir,
        # transform=transforms.Compose([ToTensor(), Normalize([.5, .5, .5], [.5, .5, .5])])
        transform=ToTensor()
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )

    # Initialize logger
    logger = TensorBoardLogger(save_dir=results_dir_root, name=results_dir_name, version=exp_name,
                               default_hp_metric=False, sub_dir="logs")

    # Initialize callbacks
    checkpointing = ModelCheckpoint(monitor="train/g_loss", save_top_k=5, mode="min")

    # Dump program arguments
    logger.log_hyperparams(params=PARAMS)

    if model == "cyclegan":
        gan_model = CycleGAN(lr_gen=lr_gen, lr_dis=lr_dis)
    elif model == "colormapgan":
        gan_model = ColorMapGAN(lr_gen=lr_gen, lr_dis=lr_dis, log_freq=log_freq)
    else:
        raise ValueError("Invalid model selection")

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpointing],
        accelerator="auto",
        devices=1,
        max_epochs=epochs,
        enable_progress_bar=enable_progress_bar
    )
    trainer.fit(gan_model, train_dataloaders=train_dataloader)


def parse_args():
    parser = argparse.ArgumentParser("Trainer for ColorMapGAN")
    parser.add_argument("--source_dir", help="Source dataset directory", required=True)
    parser.add_argument("--target_dir", help="Target dataset directory", required=True)
    parser.add_argument("--results_dir", help="Results directory", default="./results/")
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=10)
    parser.add_argument("--batch_size", help="Batch size", type=int, required=True)
    parser.add_argument("--learning_rate_gen", help="Generator learning rate", dest="lr_gen", type=float, default=0.0002)
    parser.add_argument("--learning_rate_dis", help="Generator learning rate", dest="lr_dis", type=float, default=0.0002)
    parser.add_argument("--model", help="Model name", choices=["cyclegan", "colormapgan"], default="colormapgan")
    parser.add_argument("--log_freq", help="Frequency of logging images", type=int, default=1000)
    parser.add_argument("--enable_progress_bar", help="Flag to enable progress bar", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    PARAMS = parse_args()
    main()
