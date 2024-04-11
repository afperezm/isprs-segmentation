import argparse
import pytorch_lightning as pl

from codebase.datasets.unpaired import UnpairedDataset
from codebase.models.generative import ColorMapGAN
from codebase.utils import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

NUM_CHANNELS = 3


def main():
    source_dir = PARAMS.source_dir
    target_dir = PARAMS.target_dir
    epochs = PARAMS.epochs
    batch_size = PARAMS.batch_size
    learning_rate_gen = PARAMS.learning_rate_gen
    learning_rate_dis = PARAMS.learning_rate_dis

    train_dataset = UnpairedDataset(
        source_dir=source_dir,
        target_dir=target_dir,
        transform=Compose([transforms.ToTensor()]),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )

    model = ColorMapGAN(num_classes=NUM_CHANNELS, lr_gen=learning_rate_gen, lr_dis=learning_rate_dis)
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=epochs,
    )
    trainer.fit(model, train_dataloaders=train_dataloader)


def parse_args():
    parser = argparse.ArgumentParser("Trainer for ColorMapGAN")
    parser.add_argument("--source_dir", help="Source dataset directory", required=True)
    parser.add_argument("--target_dir", help="Target dataset directory", required=True)
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=10)
    parser.add_argument("--batch_size", help="Batch size", type=int, required=True)
    parser.add_argument("--learning_rate_gen", help="Generator learning rate", type=float, default=0.0002)
    parser.add_argument("--learning_rate_dis", help="Generator learning rate", type=float, default=0.0002)
    return parser.parse_args()


if __name__ == "__main__":
    PARAMS = parse_args()
    main()
