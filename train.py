import argparse
import os
import pytorch_lightning as pl
import torch
from torchvision import transforms

from codebase.datasets import ISPRSDataset, UnpairedDataset
from codebase.models import ColorMapGAN, CycleGAN, DeepLabV3
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from time import strftime
from torch.utils.data import DataLoader, random_split


def main():
    data_dir = PARAMS.data_dir
    results_dir = PARAMS.results_dir
    epochs = PARAMS.epochs
    batch_size = PARAMS.batch_size
    learning_rate = PARAMS.learning_rate
    dataset_name = PARAMS.dataset_name
    model_name = PARAMS.model_name
    enable_progress_bar = PARAMS.enable_progress_bar
    seed = PARAMS.seed

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results_dir_root = os.path.dirname(results_dir.rstrip('/'))
    results_dir_name = os.path.basename(results_dir.rstrip('/'))

    exp_name = f"{model_name}-{strftime('%y%m%d')}-{strftime('%H%M%S')}"

    generator = torch.Generator().manual_seed(seed)

    if dataset_name == "unpaired":
        train_dataset = UnpairedDataset(
            source_dir=data_dir[0],
            target_dir=data_dir[1],
            is_train=True
        )
        # Use four images for validation and the rest for training
        valid_set_size = 4
        train_set_size = len(train_dataset) - valid_set_size
    elif dataset_name == "isprs":
        train_dataset = ISPRSDataset(
            data_dir=data_dir[0],
            is_train=True
        )
        valid_set_size = 0.2
        train_set_size = 1.0 - valid_set_size
    else:
        raise ValueError("Invalid dataset selection")

    # Split training dataset
    train_dataset, valid_dataset = random_split(train_dataset, [train_set_size, valid_set_size], generator=generator)

    if dataset_name == "unpaired":
        train_dataset.dataset.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    elif dataset_name == "isprs":
        train_dataset.dataset.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomCrop((224, 224)),
            transforms.Normalize([0.485, 0.456, 0.406, 0, 0, 0], [0.229, 0.224, 0.225, 1, 1, 1])
        ])

        valid_dataset.dataset.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406, 0, 0, 0], [0.229, 0.224, 0.225, 1, 1, 1])
        ])
    else:
        raise ValueError("Invalid dataset selection")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        generator=generator
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
    if model_name == "cyclegan" or model_name == "colormapgan":
        checkpointing = ModelCheckpoint(monitor="train/g_loss", save_top_k=5, mode="min")
    elif model_name == "deeplabv3":
        checkpointing = ModelCheckpoint(monitor="train/loss", save_top_k=5, mode="min")
    else:
        raise ValueError("Invalid model selection")

    # Dump program arguments
    logger.log_hyperparams(params=PARAMS)

    if model_name == "cyclegan":
        model = CycleGAN(lr_gen=learning_rate[0], lr_dis=learning_rate[1])
    elif model_name == "colormapgan":
        model = ColorMapGAN(lr_gen=learning_rate[0], lr_dis=learning_rate[1])
    elif model_name == "deeplabv3":
        model = DeepLabV3(num_classes=train_dataset.dataset.num_classes, learning_rate=learning_rate[0])
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
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)


def parse_args():
    parser = argparse.ArgumentParser("Trainer for ColorMapGAN")
    parser.add_argument("--data_dir", help="Source dataset directory", nargs="+", required=True)
    parser.add_argument("--results_dir", help="Results directory", default="./results/")
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=10)
    parser.add_argument("--batch_size", help="Batch size", type=int, required=True)
    parser.add_argument("--learning_rate", help="Learning rate", nargs='+', type=float, default=0.0002)
    parser.add_argument("--dataset", help="Dataset name", dest="dataset_name",
                        choices=["unpaired", "isprs"], required=True)
    parser.add_argument("--model", help="Model name", dest="model_name",
                        choices=["cyclegan", "colormapgan", "deeplabv3"], required=True)
    parser.add_argument("--enable_progress_bar", help="Flag to enable progress bar", action="store_true")
    parser.add_argument("--seed", help="Random numbers generator seed", type=int, default=42)
    parser.add_argument("--comment", help="Experiment details", default="")
    return parser.parse_args()


if __name__ == "__main__":
    PARAMS = parse_args()
    main()
