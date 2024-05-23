import argparse
import cv2
import glob
import json
import numpy as np
import os

from tqdm import tqdm

PARAMS = None


def main():
    data_dir = PARAMS.data_dir
    extract_dir = PARAMS.extract_dir
    output_dir = PARAMS.output_dir

    if not os.path.exists(os.path.join(output_dir)):
        os.makedirs(os.path.join(output_dir))

    with open(os.path.join(data_dir, "NORM_S2_patch.json"), "r") as file:
        folds_stats = json.loads(file.read())

    with open(os.path.join(extract_dir, "STATS_S2_images.json"), "r") as file:
        images_stats = json.loads(file.read())

    def parse_filename(file_path):
        return os.path.splitext(os.path.basename(file_path))[0]

    patches_paths_list = glob.glob(os.path.join(data_dir, 'DATA_S2', '*.npy'))
    patches_paths_list = [(parse_filename(patch_path).split('_')[1], patch_path) for patch_path in patches_paths_list]

    images_paths_list = glob.glob(os.path.join(extract_dir, '**', "*.png"), recursive=True)
    images_paths_dict = {parse_filename(image_path): image_path for image_path in images_paths_list}

    for patch_key, patch_path in tqdm(patches_paths_list):
        images_list = []
        patch = np.load(patch_path)
        for image_idx in range(patch.shape[0]):
            image_key = f'S2_{patch_key}_{image_idx:03d}'
            if image_key in images_paths_dict:
                # Load image and convert color
                image = cv2.imread(images_paths_dict[image_key])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # De-normalise image with image statistics
                min_vals = np.array(images_stats[image_key]['mins'])
                max_vals = np.array(images_stats[image_key]['maxs'])
                image = (max_vals - min_vals) * image / 255 + min_vals
                # De-normalize image with fold statistics
                fold_name = images_paths_dict[image_key].split(os.sep)[-4].capitalize()
                avg_vals = np.expand_dims(folds_stats[fold_name]['mean'][0:3], axis=(0, 1))
                std_vals = np.expand_dims(folds_stats[fold_name]['std'][0:3], axis=(0, 1))
                image = image * std_vals + avg_vals
            else:
                # Fall back to use non-adapted image
                image = patch[image_idx, 0:3, :, :].transpose(1, 2, 0)
            # Append de-normalised image
            images_list.append(image)
        # Store stacked images
        images_stack = np.array(images_list).transpose((0, 3, 1, 2))
        np.save(os.path.join(output_dir, f'S2_{patch_key}.npy'), images_stack)


def parse_args():
    parser = argparse.ArgumentParser("Generate PASTIS dataset from adapted images")
    parser.add_argument("--data_dir", help="Dataset directory", required=True)
    parser.add_argument("--extract_dir", help="Extracted dataset directory", required=True)
    parser.add_argument("--output_dir", help="Output directory", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    PARAMS = parse_args()
    main()
