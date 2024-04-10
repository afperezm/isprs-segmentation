import argparse
import cv2
import glob
import json
import os
import numpy as np
import progressbar
import tifffile as tiff

from sklearn.model_selection import train_test_split


def rescale_image(image, scale):
    height, width = image.shape[0], image.shape[1]

    image_rescaled = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_CUBIC)

    return image_rescaled


def pad_image(image, patch_size):

    original_height, original_width = image.shape[0], image.shape[1]
    residual_height, residual_width = original_height % patch_size, original_width % patch_size

    height_pad = 0 if residual_height == 0 else patch_size - residual_height
    width_pad = 0 if residual_width == 0 else patch_size - residual_width

    image_padded = np.pad(image, ((0, height_pad), (0, width_pad), (0, 0)), mode='constant')

    return image_padded


def crop_image_and_label(output_dir, image_path, label_path, patch_size, stride, scale):

    images_dir = os.path.join(output_dir, f'images')
    labels_dir = os.path.join(output_dir, f'labels')

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    image_filename = os.path.splitext(os.path.basename(image_path))[0]
    label_filename = os.path.splitext(os.path.basename(label_path))[0]

    image = tiff.imread(image_path)
    label = tiff.imread(label_path)

    image_rescaled = rescale_image(image, scale)
    label_rescaled = rescale_image(label, scale)

    image_padded = pad_image(image_rescaled, patch_size)
    label_padded = pad_image(label_rescaled, patch_size)

    assert image_padded.shape == label_padded.shape

    image_height, image_width = image_padded.shape[0], image_padded.shape[1]

    patches_coords = [(x, y) for y in range(0, image_height, stride) for x in range(0, image_width, stride)]

    for patch_index, (x, y) in enumerate(patches_coords, 1):
        image_patch = image_padded[y:y + patch_size, x:x + patch_size]
        label_patch = label_padded[y:y + patch_size, x:x + patch_size]

        image_patch = cv2.cvtColor(np.array(image_patch), cv2.COLOR_BGR2RGB)
        label_patch = cv2.cvtColor(np.array(label_patch), cv2.COLOR_BGR2RGB)

        out_image_path = os.path.join(images_dir, "{}_{:04}.png".format(image_filename, patch_index))
        cv2.imwrite(out_image_path, image_patch)

        out_label_path = os.path.join(labels_dir, "{}_{:04}.png".format(label_filename, patch_index))
        cv2.imwrite(out_label_path, label_patch)


def main():
    images_dir = PARAMS.images_dir
    labels_dir = PARAMS.labels_dir
    output_dir = PARAMS.output_dir
    patch_size = PARAMS.patch_size
    stride = PARAMS.stride
    scale = PARAMS.scale
    seed = PARAMS.seed

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, "params.json"), "w") as params_json:
        json.dump(vars(PARAMS), params_json, indent=4)

    images_paths = sorted(glob.glob(os.path.join(images_dir, "*.tif")))
    labels_paths = sorted(glob.glob(os.path.join(labels_dir, "*.tif")))

    num_images = len(images_paths)
    num_labels = len(labels_paths)

    assert num_images == num_labels

    train_images_paths, test_images_paths = train_test_split(images_paths, test_size=0.2, random_state=seed)
    train_labels_paths, test_labels_paths = train_test_split(labels_paths, test_size=0.2, random_state=seed)

    num_train_images = len(train_images_paths)

    bar = progressbar.ProgressBar(max_value=num_train_images)
    for idx, (img_path, msk_path) in enumerate(zip(train_images_paths, train_labels_paths)):
        crop_image_and_label(os.path.join(output_dir, 'train'), img_path, msk_path, patch_size, stride, scale)
        bar.update(idx)
    bar.update(num_train_images)

    num_test_images = len(test_images_paths)

    bar = progressbar.ProgressBar(max_value=num_test_images)
    for idx, (img_path, msk_path) in enumerate(zip(test_images_paths, test_labels_paths)):
        crop_image_and_label(os.path.join(output_dir, 'test'), img_path, msk_path, patch_size, stride, scale)
        bar.update(idx)
    bar.update(num_test_images)


def parse_args():
    parser = argparse.ArgumentParser("Spitter of ISPRS datasets into small patches")
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--labels_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    PARAMS = parse_args()
    main()
