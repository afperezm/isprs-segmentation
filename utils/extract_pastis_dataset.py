import argparse
import cv2
import numpy as np
import os

from src.dataset import PASTIS_Dataset as PASTISDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

PARAMS = None


def main():
    data_dir = PARAMS.data_dir
    output_dir = PARAMS.output_dir
    folds = PARAMS.folds

    for fold in folds:

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        dataset = PASTISDataset(folder=data_dir, folds=[fold])

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        for batch_idx, batch in enumerate(tqdm(dataloader)):

            ts = batch[0][0]
            ts_idx = dataset.id_patches[batch_idx]

            for time_idx in range(ts.shape[1]):
                image_name = f'S2_{ts_idx}_{time_idx:03d}.png'
                image_tensor = ts[0, time_idx, 0:3, :, :]
                image_array = image_tensor.numpy().transpose(1, 2, 0)
                # image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                mins = np.percentile(image_array, 0.0, axis=(0, 1), keepdims=True)
                maxs = np.percentile(image_array, 100.0, axis=(0, 1), keepdims=True)
                if np.any((maxs - mins) == 0.0):
                    continue
                image_array = 255 * (image_array - mins) / (maxs - mins)
                _ = cv2.imwrite(os.path.join(output_dir, f'fold_{fold}', image_name), image_array)


def parse_args():
    parser = argparse.ArgumentParser("Extract PASTIS dataset images")
    parser.add_argument("--data_dir", help="Dataset directory", required=True)
    parser.add_argument("--output_dir", help="Output directory", required=True)
    parser.add_argument("--folds", help="Selected fold", type=int, nargs='+', choices=range(1, 5), required=True)
    return parser.parse_args()


if __name__ == "__main__":
    PARAMS = parse_args()
    main()
