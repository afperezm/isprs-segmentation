import argparse
import os

import cv2
import numpy as np
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torchvision import transforms

from codebase.datasets.unpaired import UnpairedDataset
from codebase.models import ColorMapGAN
from codebase.models.cyclegan import CycleGAN

PARAMS = None


def main():
    data_dir = PARAMS.data_dir
    output_dir = PARAMS.output_dir
    ckpt_path = PARAMS.ckpt_path
    model_name = PARAMS.model_name
    enable_progress_bar = PARAMS.enable_progress_bar
    # test_only = PARAMS.test_only
    # predict_only = PARAMS.predict_only

    exp_name = os.path.normpath(ckpt_path).split(os.sep)[-3]

    # Create output directory
    if not os.path.exists(os.path.join(output_dir, exp_name)):
        os.makedirs(os.path.join(output_dir, exp_name))

    train_dataset = UnpairedDataset(
        source_dir=data_dir[0],
        target_dir=data_dir[0],
        is_train=True,
        include_names=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([.5, .5, .5], [.5, .5, .5])])
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8
    )

    test_dataset = UnpairedDataset(
        source_dir=data_dir[0],
        target_dir=data_dir[0],
        is_train=False,
        include_names=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([.5, .5, .5], [.5, .5, .5])])
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
    else:
        raise ValueError("Invalid model selection")

    # Initialize trainer
    trainer = pl.Trainer(logger=False, enable_progress_bar=enable_progress_bar, accelerator="auto", devices=1,
                         enable_model_summary=False)

    # Perform prediction
    results = trainer.predict(model=model, dataloaders=[train_dataloader, test_dataloader])

    # Print prediction results
    for idx, result in enumerate(results):

        image_b2a, image_b_name = result[0], result[1][0]

        print(image_b_name)

        image_b2a = np.transpose(image_b2a.cpu().detach().numpy().squeeze(), (1, 2, 0))
        image_b2a = (255 * image_b2a).astype(np.uint8)
        image_b2a = cv2.cvtColor(image_b2a, cv2.COLOR_RGB2BGR)

        print(image_b2a.shape, np.min(image_b2a), np.max(image_b2a))

        _ = cv2.imwrite(os.path.join(output_dir, exp_name, image_b_name), image_b2a)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Source dataset directory", nargs="+", required=True)
    parser.add_argument("--output_dir", help="Output directory")
    parser.add_argument("--ckpt_path", help="Checkpoint path", required=True)
    parser.add_argument("--model", help="Model name", dest="model_name", choices=["cyclegan", "colormapgan"],
                        required=True)
    parser.add_argument("--enable_progress_bar", help="Flag to enable progress bar", action="store_true")
    # parser.add_argument("--test_only", help="Flag to disable predict phase and test only", action="store_true")
    # parser.add_argument("--predict_only", help="Flag to disable test phase and predict only", action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    PARAMS = parse_args()
    main()
