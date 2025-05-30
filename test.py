import argparse
import os

import cv2
import numpy as np
import pytorch_lightning as pl
import tifffile as tiff

from torch.utils.data import DataLoader
from torchvision import transforms

from codebase.datamodules.unpaired import UnpairedDataModule
from codebase.datasets import ISPRSDataset
from codebase.datasets.flair import FLAIRDataset
from codebase.datasets.unpaired import UnpairedDataset
from codebase.models import ColorMapGAN, Segmentation
from codebase.models.cyclegan import CycleGAN
from codebase.utils.augmentation import get_validation_augmentations

PARAMS = None


def main():
    data_dir = PARAMS.data_dir
    output_dir = PARAMS.output_dir
    ckpt_path = PARAMS.ckpt_path
    dataset_name = PARAMS.dataset_name
    model_name = PARAMS.model_name
    enable_progress_bar = PARAMS.enable_progress_bar
    is_train = PARAMS.is_train
    test_only = PARAMS.test_only
    predict_only = PARAMS.predict_only

    exp_name = os.path.normpath(ckpt_path).split(os.sep)[-3]

    # Create output directories (only in prediction mode)
    if is_train and not test_only:
        os.makedirs(os.path.join(output_dir, exp_name, "train", "images"), exist_ok=True)

    if not is_train and not test_only:
        os.makedirs(os.path.join(output_dir, exp_name, "test", "images"), exist_ok=True)

    if dataset_name == "unpaired":
        test_dataset = UnpairedDataset(
            source_dir=data_dir[0],
            target_dir=data_dir[1],
            is_train=is_train,
            include_names=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([.5, .5, .5], [.5, .5, .5])])
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8
        )
    elif dataset_name == "isprs":
        test_dataset = ISPRSDataset(
            data_dir=data_dir[0],
            is_train=is_train,
            include_names=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406, 0, 0, 0], [0.229, 0.224, 0.225, 1, 1, 1])
            ])
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8
        )
    elif dataset_name == "flair":
        test_transform = get_validation_augmentations(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        test_dataset = FLAIRDataset(data_dir[0],
                                    stage='test',
                                    include_names=True,
                                    bands='rgb',
                                    transform=test_transform)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8
        )
    elif dataset_name == "unpaired-flair":
        predict_stage = 'train' if is_train else 'test'
        data_module = UnpairedDataModule(data_dir[0], batch_size=1, predict_stage=predict_stage, include_names=True,
                                         num_workers=8)
        data_module.setup(stage='predict')
        test_dataloader = data_module.predict_dataloader()
    else:
        raise ValueError("Invalid dataset selection")

    if model_name == "cyclegan":
        model = CycleGAN.load_from_checkpoint(ckpt_path)
    elif model_name == "colormapgan":
        model = ColorMapGAN.load_from_checkpoint(ckpt_path)
    elif model_name == "deeplabv3":
        model = Segmentation.load_from_checkpoint(ckpt_path, num_classes=test_dataset.num_classes,
                                                  ignore_index=test_dataset.ignore_index,
                                                  labels_palette=test_dataset.labels_palette)
    elif model_name == "deeplabv3-resnet101":
        backbone = "resnet50" if len(model_name.split('-')) == 1 else "resnet101"
        model = Segmentation.load_from_checkpoint(ckpt_path, num_classes=test_dataset.num_classes,
                                                  ignore_index=test_dataset.ignore_index,
                                                  labels_palette=test_dataset.labels_palette, backbone=backbone)
    else:
        raise ValueError("Invalid model selection")

    # Initialize trainer
    trainer = pl.Trainer(logger=False, enable_progress_bar=enable_progress_bar, accelerator="auto", devices=1,
                         enable_model_summary=False)

    if not predict_only:
        # Perform evaluation
        results = trainer.test(model=model, dataloaders=[test_dataloader])

        # Print evaluation results
        print(results)

    if not test_only:
        # Perform prediction
        results = trainer.predict(model=model, dataloaders=[test_dataloader])

        # Save predictions
        if is_train:
            split_name = "train"
        else:
            split_name = "test"

        for idx, result in enumerate(results):

            image, image_name = result[0], result[1][0]

            print(image_name)

            image = np.transpose(image.cpu().detach().numpy().squeeze(), (1, 2, 0))
            image = (255 * image).astype(np.uint8)

            print(image.shape, np.min(image), np.max(image))

            if os.path.splitext(image_name)[1] in ('.tif', '.tiff'):
                tiff.imwrite(os.path.join(output_dir, exp_name, split_name, "images", image_name), image)
            elif os.path.splitext(image_name)[1] in ('.png', '.jpg', '.jpeg'):
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                _ = cv2.imwrite(os.path.join(output_dir, exp_name, split_name, "images", image_name), image)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Source dataset directory", nargs="+", required=True)
    parser.add_argument("--output_dir", help="Output directory")
    parser.add_argument("--ckpt_path", help="Checkpoint path", required=True)
    parser.add_argument("--dataset", help="Dataset name", dest="dataset_name",
                        choices=["unpaired", "isprs", "flair", "unpaired-flair"], required=True)
    parser.add_argument("--model", help="Model name", dest="model_name",
                        choices=["cyclegan", "colormapgan", "deeplabv3", "deeplabv3-resnet101"], required=True)
    parser.add_argument("--enable_progress_bar", help="Flag to enable progress bar", action="store_true")
    parser.add_argument("--is_train", help="Flag to indicate usage of train split", action="store_true")
    parser.add_argument("--test_only", help="Flag to disable predict phase and test only", action="store_true")
    parser.add_argument("--predict_only", help="Flag to disable test phase and predict only", action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    PARAMS = parse_args()
    main()
