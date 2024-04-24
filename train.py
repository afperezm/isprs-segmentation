import argparse
import os
import pytorch_lightning as pl
import torch

from codebase.datasets.unpaired import UnpairedDataset
from codebase.models.generative import ColorMapGAN, CycleGAN
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from time import strftime
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize


def main():
    source_dir = PARAMS.source_dir
    target_dir = PARAMS.target_dir
    results_dir = PARAMS.results_dir
    epochs = PARAMS.epochs
    batch_size = PARAMS.batch_size
    lr_gen = PARAMS.lr_gen
    lr_dis = PARAMS.lr_dis
    model = PARAMS.model
    enable_progress_bar = PARAMS.enable_progress_bar
    seed = PARAMS.seed

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results_dir_root = os.path.dirname(results_dir.rstrip('/'))
    results_dir_name = os.path.basename(results_dir.rstrip('/'))

    exp_name = f"{model}-{strftime('%y%m%d')}-{strftime('%H%M%S')}"

    generator = torch.Generator().manual_seed(seed)

    train_dataset = UnpairedDataset(
        source_dir=source_dir,
        target_dir=target_dir,
        is_train=True,
        transform=Compose([ToTensor(), Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        generator=generator
    )

    valid_dataset = UnpairedDataset(
        source_dir=source_dir,
        target_dir=target_dir,
        is_train=False,
        transform=Compose([ToTensor(), Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        generator=generator
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
        gan_model = ColorMapGAN(lr_gen=lr_gen, lr_dis=lr_dis)
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
    trainer.fit(gan_model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)


def parse_args():
    parser = argparse.ArgumentParser("Trainer for ColorMapGAN")
    parser.add_argument("--source_dir", help="Source dataset directory", required=True)
    parser.add_argument("--target_dir", help="Target dataset directory", required=True)
    parser.add_argument("--results_dir", help="Results directory", default="./results/")
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=10)
    parser.add_argument("--batch_size", help="Batch size", type=int, required=True)
    parser.add_argument("--learning_rate_gen", help="Generator learning rate", dest="lr_gen", type=float,
                        default=0.0002)
    parser.add_argument("--learning_rate_dis", help="Generator learning rate", dest="lr_dis", type=float,
                        default=0.0002)
    parser.add_argument("--model", help="Model name", choices=["cyclegan", "colormapgan"], default="colormapgan")
    parser.add_argument("--enable_progress_bar", help="Flag to enable progress bar", action="store_true")
    parser.add_argument("--seed", help="Random numbers generator seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    PARAMS = parse_args()
    main()
