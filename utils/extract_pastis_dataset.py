import argparse
import cv2
import numpy as np
import os

from src.dataset import PASTIS_Dataset as PASTISDataset
from torch.utils.data import DataLoader

PARAMS = None


def main():
    data_dir = PARAMS.data_dir
    output_dir = PARAMS.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = PASTISDataset(folder=data_dir, folds=[1])

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for batch_idx, batch in enumerate(dataloader):

        ts = batch[0][0]
        ts_idx = dataset.id_patches[batch_idx]

        for time_idx in range(ts.shape[1]):
            image_name = f'S2_{ts_idx}_{time_idx:03d}.png'
            image_array = ts[:, time_idx, 0:3, :, :].numpy().transpose(1, 2, 0)
            print(image_name, image_array.shape, np.min(image_array), np.max(image_array))
            _ = cv2.imwrite(os.path.join(output_dir, image_name), image_array)


def parse_args():
    parser = argparse.ArgumentParser("Extract PASTIS dataset images")
    parser.add_argument("--data_dir", help="Dataset directory", required=True)
    parser.add_argument("--output_dir", help="Output directory", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    PARAMS = parse_args()
    main()
