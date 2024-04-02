import argparse
import cv2
import glob
import os
import numpy as np
import progressbar
import tifffile as tiff


IMAGES_RGB_DIR = '2_Ortho_RGB'
IMAGES_IR_DIR = '3_Ortho_IRRG'
LABELS_DIR = '5_Labels_all'


def pad_image(image, patch_size):

    original_height, original_width = image.shape[0], image.shape[1]
    residual_height, residual_width = original_height % patch_size, original_width % patch_size

    width_pad = 0 if residual_width == 0 else patch_size - residual_width
    height_pad = 0 if residual_height == 0 else patch_size - residual_height

    image_padded = np.pad(image, ((0, height_pad), (0, width_pad), (0, 0)), mode='constant')

    img_pad = image_padded['image']

    return img_pad


def spit_image_and_label(images_dir, labels_dir, image_path, label_path, patch_size, stride):
    image_filename = os.path.splitext(os.path.basename(image_path))[0]
    label_filename = os.path.splitext(os.path.basename(label_path))[0]

    image = tiff.imread(image_path)
    label = tiff.imread(label_path)

    image_padded = pad_image(image, patch_size)
    label_padded = pad_image(label, patch_size)

    assert image_padded.shape == label_padded.shape

    image_height, image_width = image_padded.shape[0], image_padded.shape[1]

    patches_coords = [(x, y) for y in range(0, image_height, stride) for x in range(0, image_width, stride)]

    for patch_index, (x, y) in enumerate(patches_coords):
        image_patch = image_padded[y:y + patch_size, x:x + patch_size]
        label_patch = label_padded[y:y + patch_size, x:x + patch_size]

        out_image_path = os.path.join(images_dir, "{}_{}.png".format(image_filename, patch_index))
        cv2.imwrite(out_image_path, image_patch)

        out_label_path = os.path.join(labels_dir, "{}_{}.png".format(label_filename, patch_index))
        cv2.imwrite(out_label_path, label_patch)


def main():
    data_dir = PARAMS.data_dir
    output_dir = PARAMS.output_dir
    use_ir = PARAMS.use_ir
    patch_size = PARAMS.patch_size
    stride = PARAMS.stride

    if use_ir:
        images_dir = os.path.join(data_dir, IMAGES_IR_DIR)
    else:
        images_dir = os.path.join(data_dir, IMAGES_RGB_DIR)

    labels_dir = os.path.join(data_dir, LABELS_DIR)

    if use_ir:
        mode = "irrg"
    else:
        mode = "rgb"

    output_images_dir = os.path.join(output_dir, f'images_{mode}_{patch_size}_{stride}')
    output_labels_dir = os.path.join(output_dir, f'labels_{patch_size}_{stride}')

    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)

    if not os.path.exists(output_labels_dir):
        os.makedirs(output_labels_dir)

    images_paths = sorted(glob.glob(os.path.join(images_dir, "*.tif")))
    labels_paths = sorted(glob.glob(os.path.join(labels_dir, "*.tif")))

    num_images = len(images_paths)
    num_labels = len(labels_paths)

    assert num_images == num_labels

    bar = progressbar.ProgressBar(max_value=num_images)
    for idx, (img_path, msk_path) in enumerate(zip(images_paths, labels_paths)):
        spit_image_and_label(output_images_dir, output_labels_dir, img_path, msk_path, patch_size, stride)
        bar.update(idx)
    bar.update(num_images)


def parse_args():
    parser = argparse.ArgumentParser("Spitter of ISPRS datasets into small patches")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--use_ir", action="store_true")
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    PARAMS = parse_args()
    main()
