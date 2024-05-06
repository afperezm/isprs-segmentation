import argparse
import os

import cv2
import numpy as np
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torchvision import transforms

from codebase.datasets import ISPRSDataset
from codebase.datasets.unpaired import UnpairedDataset
from codebase.models import ColorMapGAN, DeepLabV3
from codebase.models.cyclegan import CycleGAN

PARAMS = None


def main():
    data_dir = PARAMS.data_dir
    output_dir = PARAMS.output_dir
    ckpt_path = PARAMS.ckpt_path
    dataset_name = PARAMS.dataset_name
    model_name = PARAMS.model_name
    enable_progress_bar = PARAMS.enable_progress_bar
    test_only = PARAMS.test_only
    predict_only = PARAMS.predict_only

    exp_name = os.path.normpath(ckpt_path).split(os.sep)[-3]

    # Create output directories
    os.makedirs(os.path.join(output_dir, exp_name, "train", "images"))
    os.makedirs(os.path.join(output_dir, exp_name, "test", "images"))

    if dataset_name == "unpaired":
        train_dataset = UnpairedDataset(
            source_dir=data_dir[0],
            target_dir=data_dir[1],
            is_train=True,
            include_names=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([.5, .5, .5], [.5, .5, .5])])
        )

        test_dataset = UnpairedDataset(
            source_dir=data_dir[0],
            target_dir=data_dir[1],
            is_train=False,
            include_names=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([.5, .5, .5], [.5, .5, .5])])
        )
    elif dataset_name == "isprs":
        train_dataset = ISPRSDataset(
            data_dir=data_dir,
            is_train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406, 0, 0, 0], [0.229, 0.224, 0.225, 1, 1, 1])
            ])
        )

        test_dataset = ISPRSDataset(
            data_dir=data_dir,
            is_train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406, 0, 0, 0], [0.229, 0.224, 0.225, 1, 1, 1])
            ])
        )
    else:
        raise ValueError("Invalid dataset selection")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8
    )

    if model_name == "cyclegan":
        model = CycleGAN.load_from_checkpoint(ckpt_path)
    elif model_name == "colormapgan":
        model = ColorMapGAN.load_from_checkpoint(ckpt_path)
    elif model_name == "deeplabv3":
        model = DeepLabV3.load_from_checkpoint(ckpt_path, num_classes=train_dataset.num_classes)
    else:
        raise ValueError("Invalid model selection")

    # Initialize trainer
    trainer = pl.Trainer(logger=False, enable_progress_bar=enable_progress_bar, accelerator="auto", devices=1,
                         enable_model_summary=False)

    if not predict_only:
        # Perform evaluation
        results = trainer.test(model=model, dataloaders=[train_dataloader, test_dataloader])

        # Print evaluation results
        print(results)

    if not test_only:
        # Perform prediction
        results = trainer.predict(model=model, dataloaders=[train_dataloader, test_dataloader])

        assert len(results) == 2

        # Save predictions
        for split_idx, split_results in enumerate(results):

            if split_idx == 0:
                split_name = "train"
            else:
                split_name = "test"

            for idx, result in enumerate(split_results):

                image_b2a, image_b_name = result[0], result[1][0]

                print(image_b_name)

                image_b2a = np.transpose(image_b2a.cpu().detach().numpy().squeeze(), (1, 2, 0))
                image_b2a = (255 * image_b2a).astype(np.uint8)
                image_b2a = cv2.cvtColor(image_b2a, cv2.COLOR_RGB2BGR)

                print(image_b2a.shape, np.min(image_b2a), np.max(image_b2a))

                _ = cv2.imwrite(os.path.join(output_dir, exp_name, split_name, "images", image_b_name), image_b2a)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Source dataset directory", nargs="+", required=True)
    parser.add_argument("--output_dir", help="Output directory")
    parser.add_argument("--ckpt_path", help="Checkpoint path", required=True)
    parser.add_argument("--dataset", help="Dataset name", dest="dataset_name",
                        choices=["unpaired", "isprs"], required=True)
    parser.add_argument("--model", help="Model name", dest="model_name",
                        choices=["cyclegan", "colormapgan", "deeplabv3"], required=True)
    parser.add_argument("--enable_progress_bar", help="Flag to enable progress bar", action="store_true")
    parser.add_argument("--test_only", help="Flag to disable predict phase and test only", action="store_true")
    parser.add_argument("--predict_only", help="Flag to disable test phase and predict only", action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    PARAMS = parse_args()
    main()
