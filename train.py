import argparse
import os
import pytorch_lightning as pl
import torch
from torchvision import transforms

from codebase.datamodules.unpaired import FLAIRDataModule
from codebase.datasets import ISPRSDataset, UnpairedDataset
from codebase.datasets.flair import FLAIRDataset
from codebase.models import ColorMapGAN, CycleGAN, DeepLabV3
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from time import strftime
from torch.utils.data import DataLoader, random_split

from codebase.utils.augmentation import choose_training_augmentations, get_validation_augmentations


def main():
    data_dir = PARAMS.data_dir
    results_dir = PARAMS.results_dir
    epochs = PARAMS.epochs
    batch_size = PARAMS.batch_size
    learning_rate = PARAMS.learning_rate
    weight_decay = PARAMS.weight_decay
    lambdas = PARAMS.lambdas
    dataset_name = PARAMS.dataset_name
    model_name = PARAMS.model_name
    valid_size = PARAMS.valid_size
    scheduler_factor = PARAMS.scheduler_factor
    scheduler_patience = PARAMS.scheduler_patience
    scheduler_threshold = PARAMS.scheduler_threshold
    enable_progress_bar = PARAMS.enable_progress_bar
    seed = PARAMS.seed
    ckpt_path = PARAMS.ckpt_path
    resume = PARAMS.resume

    torch.set_float32_matmul_precision('highest')

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results_dir_root = os.path.dirname(results_dir.rstrip('/'))
    results_dir_name = os.path.basename(results_dir.rstrip('/'))

    if resume:
        exp_name = os.path.normpath(ckpt_path).split(os.sep)[-3]
    else:
        exp_name = f"{model_name}-{strftime('%y%m%d')}-{strftime('%H%M%S')}"

    generator = torch.Generator().manual_seed(seed)

    if dataset_name == "unpaired":
        train_dataset = UnpairedDataset(
            source_dir=data_dir[0],
            target_dir=data_dir[1],
            is_train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        )
        # Split training dataset
        train_size = 1.0 - valid_size
        train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size], generator=generator)

        valid_batch_size = 4 if batch_size == 1 else batch_size
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                      generator=generator)
        valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False, num_workers=8,
                                      generator=generator)
    elif dataset_name == "unpaired-pastis":
        train_dataset = UnpairedDataset(
            source_dir=data_dir[0],
            target_dir=data_dir[1],
            is_train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
            ])
        )
        # Split training dataset
        train_size = 1.0 - valid_size
        train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size], generator=generator)

        valid_batch_size = 4 if batch_size == 1 else batch_size
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                      generator=generator)
        valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False, num_workers=8,
                                      generator=generator)
    elif dataset_name == "isprs":
        train_dataset = ISPRSDataset(
            data_dir=data_dir[0],
            is_train=True
        )
        # Split training dataset
        train_size = 1.0 - valid_size
        train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size], generator=generator)
        # Assign training transform
        train_dataset.dataset.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomCrop((224, 224)),
            transforms.Normalize([0.485, 0.456, 0.406, 0, 0, 0], [0.229, 0.224, 0.225, 1, 1, 1])
        ])
        # Assign validation transform
        valid_dataset.dataset.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406, 0, 0, 0], [0.229, 0.224, 0.225, 1, 1, 1])
        ])
        valid_batch_size = batch_size
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                      generator=generator)
        valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False, num_workers=8,
                                      generator=generator)
    elif dataset_name == "flair":
        train_dataset = FLAIRDataset(data_dir[0],
                                     os.path.join(data_dir[0], 'sub_train_imgs.txt'),
                                     os.path.join(data_dir[0], 'sub_train_masks.txt'),
                                     bands='rgb')
        # Split training dataset
        train_size = 1.0 - valid_size
        train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size], generator=generator)
        # Assign training transform
        train_dataset.dataset.transform = choose_training_augmentations(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225],
                                                                        aug_type='randaugment')
        # Assign validation transform
        valid_dataset.dataset.transform = get_validation_augmentations(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225])
        valid_batch_size = batch_size
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                      generator=generator)
        valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False, num_workers=8,
                                      generator=generator)
    elif dataset_name == "unpaired-flair":
        data_module = FLAIRDataModule(data_dir[0], batch_size=batch_size, num_workers=8, generator=generator)
        data_module.setup(stage='fit')
        train_dataloader = data_module.train_dataloader()
        valid_dataloader = data_module.val_dataloader()
    else:
        raise ValueError("Invalid dataset selection")

    # Initialize logger
    tb_logger = TensorBoardLogger(save_dir=results_dir_root, name=results_dir_name, version=exp_name,
                                  default_hp_metric=False, sub_dir="logs")
    wb_logger = WandbLogger(name=exp_name, project="UDA for Remote Sensing Image Semantic Segmentation")

    # Initialize callbacks
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    if model_name == "cyclegan" or model_name == "colormapgan":
        checkpointing = ModelCheckpoint(monitor="train/g_loss", save_last=True, save_top_k=5, mode="min")
    elif model_name == "deeplabv3" or model_name == "deeplabv3-resnet101":
        checkpointing = ModelCheckpoint(monitor="valid/loss", save_last=True, save_top_k=5, mode="min")
    else:
        raise ValueError("Invalid model selection")

    # Dump program arguments
    tb_logger.log_hyperparams(params=PARAMS)
    wb_logger.log_hyperparams(params=PARAMS)

    if model_name == "cyclegan":
        model = CycleGAN(lr_gen=learning_rate[0], lr_dis=learning_rate[1])
    elif model_name == "colormapgan":
        model = ColorMapGAN(lr_gen=learning_rate[0], lr_dis=learning_rate[1])
    elif model_name == "deeplabv3" or model_name == "deeplabv3-resnet101":
        backbone = "resnet50" if len(model_name.split('-')) == 1 else "resnet101"
        if ckpt_path and not resume:
            model = DeepLabV3.load_from_checkpoint(ckpt_path, num_classes=train_dataset.dataset.num_classes,
                                                   ignore_index=train_dataset.dataset.ignore_index,
                                                   labels_palette=train_dataset.dataset.labels_palette,
                                                   backbone=backbone,
                                                   loss_ce_weight=lambdas[0], loss_dice_weight=lambdas[1],
                                                   backbone_learning_rate=learning_rate[0],
                                                   classifier_learning_rate=learning_rate[1],
                                                   backbone_weight_decay=weight_decay[0],
                                                   classifier_weight_decay=weight_decay[1],
                                                   scheduler_factor=scheduler_factor,
                                                   scheduler_patience=scheduler_patience,
                                                   scheduler_threshold=scheduler_threshold)
        else:
            model = DeepLabV3(num_classes=train_dataset.dataset.num_classes, backbone=backbone,
                              ignore_index=train_dataset.dataset.ignore_index,
                              labels_palette=train_dataset.dataset.labels_palette,
                              loss_ce_weight=lambdas[0], loss_dice_weight=lambdas[1],
                              backbone_learning_rate=learning_rate[0], classifier_learning_rate=learning_rate[1],
                              backbone_weight_decay=weight_decay[0], classifier_weight_decay=weight_decay[1],
                              scheduler_factor=scheduler_factor, scheduler_patience=scheduler_patience,
                              scheduler_threshold=scheduler_threshold)
    else:
        raise ValueError("Invalid model selection")

    # Initialize trainer
    trainer = pl.Trainer(
        logger=[tb_logger, wb_logger],
        callbacks=[lr_monitor, checkpointing],
        accelerator="auto",
        devices=1,
        max_epochs=epochs,
        enable_progress_bar=enable_progress_bar
    )

    # Perform training
    if resume:
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)


def parse_args():
    parser = argparse.ArgumentParser("Generic trainer for UDA adaptation/segmentation")
    parser.add_argument("--data_dir", help="Source dataset directory", nargs="+", required=True)
    parser.add_argument("--results_dir", help="Results directory", default="./results/")
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=10)
    parser.add_argument("--batch_size", help="Batch size", type=int, required=True)
    parser.add_argument("--learning_rate", help="Learning rate", nargs='+', type=float, default=[0.0002, 0.0002])
    parser.add_argument("--weight_decay", help="Weight_decay", nargs='+', type=float, default=[0.0, 0.0])
    parser.add_argument("--lambdas", help="Losses weights", type=float, nargs="+", default=[1.0, 0.0])
    parser.add_argument("--dataset", help="Dataset name", dest="dataset_name",
                        choices=["unpaired", "unpaired-pastis", "isprs", "flair", "unpaired-flair"], required=True)
    parser.add_argument("--model", help="Model name", dest="model_name",
                        choices=["cyclegan", "colormapgan", "deeplabv3", "deeplabv3-resnet101"], required=True)
    parser.add_argument("--valid_size", help="Validation dataset size", type=float, default=0.2)
    parser.add_argument("--scheduler_factor", help="LR Scheduler reduction factor", type=float, default=0.2)
    parser.add_argument("--scheduler_patience", help="Number of epochs with improvement", type=int, default=7)
    parser.add_argument("--scheduler_threshold", help="Threshold to measure improvement", type=float, default=0.0001)
    parser.add_argument("--enable_progress_bar", help="Flag to enable progress bar", action="store_true")
    parser.add_argument("--seed", help="Random numbers generator seed", type=int, default=42)
    parser.add_argument("--ckpt_path", help="Resume checkpoint path")
    parser.add_argument("--resume", help="Resume training", action="store_true")
    parser.add_argument("--comment", help="Experiment details", default="")
    return parser.parse_args()


if __name__ == "__main__":
    PARAMS = parse_args()
    main()
