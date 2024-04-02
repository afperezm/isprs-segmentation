# import albumentations
import argparse
import cv2
import glob
import os
import numpy as np
import progressbar
import tifffile as tiff


# import multiprocessing as mp
# import random
# import time

# import torch

# from multiprocessing import cpu_count
# from multiprocessing.pool import Pool
# from torchvision.transforms import (Pad, ColorJitter, Resize, FiveCrop, RandomResizedCrop,
#                                     RandomHorizontalFlip, RandomRotation, RandomVerticalFlip)
# from PIL import Image

# NUM_CLASSES = 6
# SEED = 42

# IMAGES_RGB_DIR = '2_Ortho_RGB'
# IMAGES_IRRG_DIR = '3_Ortho_IRRG'

IMAGES_DIR = '2_Ortho_RGB'
LABELS_DIR = '5_Labels_all'


# def seed_everything(seed):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = True


# ImSurf = np.array([255, 255, 255])  # label 0
# BUILDING = np.array([255, 0, 0])  # label 1
# LOW_VEGETATION = np.array([255, 255, 0])  # label 2
# TREE = np.array([0, 255, 0])  # label 3
# CAR = np.array([0, 255, 255])  # label 4
# CLUTTER = np.array([0, 0, 255])  # label 5
# BOUNDARY = np.array([0, 0, 0])  # label 6


# def get_img_mask_padded(image, mask, patch_size):
#     img, mask = np.array(image), np.array(mask)
#     oh, ow = img.shape[0], img.shape[1]
#     rh, rw = oh % patch_size, ow % patch_size
#     width_pad = 0 if rw == 0 else patch_size - rw
#     height_pad = 0 if rh == 0 else patch_size - rh
#
#     h, w = oh + height_pad, ow + width_pad
#
#     pad_img = albu.PadIfNeeded(min_height=h, min_width=w, position='bottom_right', border_mode=cv2.BORDER_CONSTANT,
#                                value=0)(image=img)
#     pad_mask = albu.PadIfNeeded(min_height=h, min_width=w, position='bottom_right', border_mode=cv2.BORDER_CONSTANT,
#                                 value=6)(image=mask)
#
#     img_pad, mask_pad = pad_img['image'], pad_mask['image']
#     img_pad = cv2.cvtColor(np.array(img_pad), cv2.COLOR_RGB2BGR)
#     mask_pad = cv2.cvtColor(np.array(mask_pad), cv2.COLOR_RGB2BGR)
#     return img_pad, mask_pad


def pad_image(image, patch_size):
    # image_array = np.array(image)

    original_height, original_width = image.shape[0], image.shape[1]
    residual_height, residual_width = original_height % patch_size, original_width % patch_size

    width_pad = 0 if residual_width == 0 else patch_size - residual_width
    height_pad = 0 if residual_height == 0 else patch_size - residual_height

    # height, width = original_height + height_pad, original_width + width_pad

    # image_padded = albumentations.PadIfNeeded(min_height=height, min_width=width, position='top_left',
    #                                           border_mode=cv2.BORDER_CONSTANT, value=0)(image=image)

    image_padded = np.pad(image, ((0, height_pad), (0, width_pad), (0, 0)), mode='constant')

    img_pad = image_padded['image']
    # img_pad = cv2.cvtColor(np.array(img_pad), cv2.COLOR_BGR2RGB)

    return img_pad


# def pv2rgb(mask):
#     h, w = mask.shape[0], mask.shape[1]
#     mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
#     mask_convert = mask[np.newaxis, :, :]
#     mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 255, 0]
#     mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
#     mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
#     mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
#     mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 204, 255]
#     mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 0, 255]
#     return mask_rgb


# def car_color_replace(mask):
#     mask = cv2.cvtColor(np.array(mask.copy()), cv2.COLOR_RGB2BGR)
#     mask[np.all(mask == [0, 255, 255], axis=-1)] = [0, 204, 255]
#     return mask


# def image_to_label(_label):
#     _label = _label.transpose(2, 0, 1)
#     label_seg = np.zeros(_label.shape[1:], dtype=np.uint8)
#     label_seg[np.all(_label.transpose([1, 2, 0]) == ImSurf, axis=-1)] = 0
#     label_seg[np.all(_label.transpose([1, 2, 0]) == BUILDING, axis=-1)] = 1
#     label_seg[np.all(_label.transpose([1, 2, 0]) == LOW_VEGETATION, axis=-1)] = 2
#     label_seg[np.all(_label.transpose([1, 2, 0]) == TREE, axis=-1)] = 3
#     label_seg[np.all(_label.transpose([1, 2, 0]) == CAR, axis=-1)] = 4
#     label_seg[np.all(_label.transpose([1, 2, 0]) == CLUTTER, axis=-1)] = 5
#     label_seg[np.all(_label.transpose([1, 2, 0]) == BOUNDARY, axis=-1)] = 6
#     return label_seg


# def image_augment(image, label, patch_size):
#     image_list = []
#     mask_list = []
#
#     image_height, image_width = image.shape[0], image.shape[1]
#     label_height, label_width = label.shape[0], label.shape[1]
#
#     assert image_height == label_height and image_width == label_width
#
#     # if mode == 'train':
#     #     h_vlip = RandomHorizontalFlip(p=1.0)
#     #     v_vlip = RandomVerticalFlip(p=1.0)
#     #     image_h_vlip, mask_h_vlip = h_vlip(image.copy()), h_vlip(label.copy())
#     #     image_v_vlip, mask_v_vlip = v_vlip(image.copy()), v_vlip(label.copy())
#     #
#     #     image_list_train = [image, image_h_vlip, image_v_vlip]
#     #     mask_list_train = [label, mask_h_vlip, mask_v_vlip]
#     #     for i in range(len(image_list_train)):
#     #         image_tmp, mask_tmp = get_img_mask_padded(image_list_train[i], mask_list_train[i], patch_size)
#     #         mask_tmp = rgb_to_2D_label(mask_tmp.copy())
#     #         image_list.append(image_tmp)
#     #         mask_list.append(mask_tmp)
#     # else:
#     #     rescale = Resize(size=(int(image_width * val_scale), int(image_height * val_scale)))
#     #     image, label = rescale(image.copy()), rescale(label.copy())
#     #     image, label = get_img_mask_padded(image.copy(), label.copy(), patch_size)
#     #     label = rgb_to_2D_label(label.copy())
#     #
#     #     image_list.append(image)
#     #     mask_list.append(label)
#
#     image_padded = pad_image(image, patch_size)
#     label_padded = pad_image(label, patch_size)
#
#     # label = image_to_label(label.copy())
#
#     image_list.append(image_padded)
#     mask_list.append(label_padded)
#
#     return image_list, mask_list


# def car_aug(image, mask):
#     assert image.shape[:2] == mask.shape
#     resize_crop_1 = albu.Compose([albu.Resize(width=int(image.shape[0] * 1.25), height=int(image.shape[1] * 1.25)),
#                                   albu.RandomCrop(width=image.shape[0], height=image.shape[1])])(image=image.copy(),
#                                                                                                  mask=mask.copy())
#     resize_crop_2 = albu.Compose([albu.Resize(width=int(image.shape[0] * 1.5), height=int(image.shape[1] * 1.5)),
#                                   albu.RandomCrop(width=image.shape[0], height=image.shape[1])])(image=image.copy(),
#                                                                                                  mask=mask.copy())
#     resize_crop_3 = albu.Compose([albu.Resize(width=int(image.shape[0] * 1.75), height=int(image.shape[1] * 1.75)),
#                                   albu.RandomCrop(width=image.shape[0], height=image.shape[1])])(image=image.copy(),
#                                                                                                  mask=mask.copy())
#     resize_crop_4 = albu.Compose([albu.Resize(width=int(image.shape[0] * 2.0), height=int(image.shape[1] * 2.0)),
#                                   albu.RandomCrop(width=image.shape[0], height=image.shape[1])])(image=image.copy(),
#                                                                                                  mask=mask.copy())
#     v_flip = albu.VerticalFlip(p=1.0)(image=image.copy(), mask=mask.copy())
#     h_flip = albu.HorizontalFlip(p=1.0)(image=image.copy(), mask=mask.copy())
#     rotate_90 = albu.RandomRotate90(p=1.0)(image=image.copy(), mask=mask.copy())
#     image_resize_crop_1, mask_resize_crop_1 = resize_crop_1['image'], resize_crop_1['mask']
#     image_resize_crop_2, mask_resize_crop_2 = resize_crop_2['image'], resize_crop_2['mask']
#     image_resize_crop_3, mask_resize_crop_3 = resize_crop_3['image'], resize_crop_3['mask']
#     image_resize_crop_4, mask_resize_crop_4 = resize_crop_4['image'], resize_crop_4['mask']
#     image_vflip, mask_vflip = v_flip['image'], v_flip['mask']
#     image_hflip, mask_hflip = h_flip['image'], h_flip['mask']
#     image_rotate, mask_rotate = rotate_90['image'], rotate_90['mask']
#     image_list = [image, image_resize_crop_1, image_resize_crop_2, image_resize_crop_3,
#                   image_resize_crop_4, image_vflip, image_hflip, image_rotate]
#     mask_list = [mask, mask_resize_crop_1, mask_resize_crop_2, mask_resize_crop_3,
#                  mask_resize_crop_4, mask_vflip, mask_hflip, mask_rotate]
#
#     return image_list, mask_list


# def patch_format(inp):
# (img_path, mask_path, imgs_output_dir, masks_output_dir, patch_size, stride) = inp

def spit_image_and_label(images_dir, labels_dir, image_path, label_path, patch_size, stride):
    image_filename = os.path.splitext(os.path.basename(image_path))[0]
    label_filename = os.path.splitext(os.path.basename(label_path))[0]

    # image = Image.open(image_path).convert('RGB')
    image = tiff.imread(image_path)
    # label = Image.open(label_path).convert('RGB')
    label = tiff.imread(label_path)

    # if gt:
    #     mask_ = car_color_replace(label.copy())
    #     out_origin_mask_path = os.path.join(masks_output_dir + '/origin/', "{}.tif".format(msk_filename))
    #     cv2.imwrite(out_origin_mask_path, mask_)

    # # print(mask)
    # # print(img_path)
    # # print(img.size, mask.size)
    # # img and mask shape: WxHxC
    # image_list, mask_list = image_augment(image=image, label=label, patch_size=patch_size)

    image_padded = pad_image(image, patch_size)
    label_padded = pad_image(label, patch_size)

    # assert len(image_list) == len(mask_list) and image_list[0].shape == mask_list[0].shape

    # for m in range(len(image_list)):
    #
    #     k = 0
    #
    #     image = image_list[m]
    #     label = mask_list[m]

    assert image_padded.shape == label_padded.shape

    image_height, image_width = image_padded.shape[0], image_padded.shape[1]
    # label_height, label_width = label.shape[0], label.shape[1]

    # if gt:
    #     label = pv2rgb(label.copy())

    patches_coords = [(x, y) for y in range(0, image_height, stride) for x in range(0, image_width, stride)]

    for patch_index, (x, y) in enumerate(patches_coords):
        image_patch = image_padded[y:y + patch_size, x:x + patch_size]
        label_patch = label_padded[y:y + patch_size, x:x + patch_size]

        # img_tile, mask_tile = image_patch, label_patch

        # if img_tile.shape[0] == patch_size and img_tile.shape[1] == patch_size \
        #         and mask_tile.shape[0] == patch_size and mask_tile.shape[1] == patch_size:
        #
        #     bins = np.array(range(NUM_CLASSES + 1))
        #     class_pixel_counts, _ = np.histogram(mask_tile, bins=bins)
        #     cf = class_pixel_counts / (mask_tile.shape[0] * mask_tile.shape[1])
        #
        #     if cf[4] > 1.0 and mode == 'train':  # ignore car_aug, no improvement
        #         car_imgs, car_masks = car_aug(img_tile, mask_tile)
        #         for i in range(len(car_imgs)):
        #             out_img_path = os.path.join(imgs_output_dir,
        #                                         "{}_{}_{}_{}.tif".format(img_filename, m, k, i))
        #             cv2.imwrite(out_img_path, car_imgs[i])
        #
        #             out_mask_path = os.path.join(masks_output_dir,
        #                                          "{}_{}_{}_{}.png".format(msk_filename, m, k, i))
        #             cv2.imwrite(out_mask_path, car_masks[i])
        #     else:
        #         out_img_path = os.path.join(imgs_output_dir, "{}_{}_{}.tif".format(img_filename, m, k))
        #         cv2.imwrite(out_img_path, img_tile)
        #
        #         out_mask_path = os.path.join(masks_output_dir, "{}_{}_{}.png".format(msk_filename, m, k))
        #         cv2.imwrite(out_mask_path, mask_tile)

        out_image_path = os.path.join(images_dir, "{}_{}.png".format(image_filename, patch_index))
        cv2.imwrite(out_image_path, image_patch)

        out_label_path = os.path.join(labels_dir, "{}_{}.png".format(label_filename, patch_index))
        cv2.imwrite(out_label_path, label_patch)

        # k += 1


def main():
    # images_dir = PARAMS.images_dir
    # labels_dir = PARAMS.masks_dir
    data_dir = PARAMS.data_dir
    # output_images_dir = PARAMS.output_images_dir
    # output_labels_dir = PARAMS.output_labels_dir
    output_dir = PARAMS.output_dir
    # gt = args.gt
    # mode = args.mode
    # val_scale = PARAMS.val_scale
    patch_size = PARAMS.patch_size
    stride = PARAMS.stride_size

    images_dir = os.path.join(data_dir, IMAGES_DIR)
    labels_dir = os.path.join(data_dir, LABELS_DIR)

    output_images_dir = os.path.join(output_dir, f'images_{patch_size}_{stride}')
    output_labels_dir = os.path.join(output_dir, f'labels_{patch_size}_{stride}')

    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)

    if not os.path.exists(output_labels_dir):
        os.makedirs(output_labels_dir)
        # if gt:
        #     os.makedirs(output_labels_dir + '/origin')

    # seed_everything(SEED)

    images_paths = sorted(glob.glob(os.path.join(images_dir, "*.tif")))
    labels_paths = sorted(glob.glob(os.path.join(labels_dir, "*.tif")))

    num_images = len(images_paths)
    num_labels = len(labels_paths)

    assert num_images == num_labels

    # inp = [(img_path, msk_path, output_images_dir, output_labels_dir, patch_size, stride_size) for
    #        img_path, msk_path in zip(img_paths, msk_paths)]
    #
    # t0 = time.time()
    # pool.Pool(processes=mp.cpu_count()).map(patch_format, inp)
    # t1 = time.time()
    # split_time = t1 - t0
    # print('images splitting spends: {} s'.format(split_time))

    bar = progressbar.ProgressBar(max_value=num_images)
    for idx, (img_path, msk_path) in enumerate(zip(images_paths, labels_paths)):
        spit_image_and_label(output_images_dir, output_labels_dir, img_path, msk_path, patch_size, stride)
        bar.update(idx)
    bar.update(num_images)


def parse_args():
    parser = argparse.ArgumentParser("Spitter of ISPRS datasets into small patches")
    parser.add_argument("--data_dir", required=True)
    # parser.add_argument("--images_dir", required=True)  # data/potsdam/train_images
    # parser.add_argument("--masks_dir", required=True)  # data/potsdam/train_masks
    parser.add_argument("--output_dir", required=True)
    # parser.add_argument("--output_images_dir", required=True)  # data/potsdam/train/images_1024
    # parser.add_argument("--output_labels_dir", required=True)  # data/potsdam/train/masks_1024
    # parser.add_argument("--gt", action='store_true')  # output RGB mask
    # parser.add_argument("--mode", type=str, default='train')
    # parser.add_argument("--val_scale", type=float, default=1.0)  # ignore
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    PARAMS = parse_args()
    main()
