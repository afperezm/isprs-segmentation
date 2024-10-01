import sys

import cv2
import numpy as np
import tifffile as tiff
import os
import random
import torch

from torch.utils.data import Dataset, DataLoader


class FLAIRDataset(Dataset):

    ignore_index = 0

    label_mapping = {
        0: [0, 0, 0],  # other
        1: [219, 14, 154],  # building
        2: [147, 142, 123],  # pervious surface
        3: [248, 12, 0],  # impervious surface
        4: [169, 113, 1],  # bare soil
        5: [21, 83, 174],  # water
        6: [25, 74, 38],  # coniferous
        7: [70, 228, 131],  # deciduous
        8: [243, 166, 13],  # brushwood
        9: [102, 0, 130],  # vineyard
        10: [85, 255, 0],  # herbaceous vegetation
        11: [255, 243, 13],  # agricultural land
        12: [228, 223, 124],  # plowed land
    }

    def __init__(
            self,
            data_dir,
            bands='rgbirh',
            transform=None,
            crop_size=None,
            geo_info=None,
            stage='train',
            include_names=False
    ):

        self.data_dir = data_dir
        self.bands = bands
        self.transform = transform
        self.crop_size = crop_size
        self.geo_info = geo_info
        self.stage = stage
        self.include_names = include_names

        images_txt = os.path.join(self.data_dir, f'sub_{self.stage}_imgs.txt')

        with open(images_txt) as f:
            lines = f.readlines()
        self.images_fps = sorted([line.strip() for line in lines])

        masks_txt = os.path.join(self.data_dir, f'sub_{self.stage}_masks.txt')

        with open(masks_txt) as f:
            lines = f.readlines()
        self.masks_fps = sorted([line.strip() for line in lines])

    def load_bands(self, img):
        if self.bands == 'rgbirh':
            return img
        elif self.bands == 'rgb':
            return img[:, :, :3]
        elif self.bands == 'rgbir':
            return img[:, :, :4]
        else:
            return img

    def crop_or_resize(self, image, mask):
        n = random.randint(1, 2)
        if n == 1:
            choice = self.random_crop(image, mask)
        else:
            choice = self.im_resize(image, mask)
        return choice

    def im_resize(self, image, mask):
        image = cv2.resize(image, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)
        return image, mask

    def random_crop(self, image, mask):
        h = np.random.randint(0, self.crop_size)
        w = np.random.randint(0, self.crop_size)
        image = image[h:h + self.crop_size, w:w + self.crop_size, :]
        mask = mask[h:h + self.crop_size, w:w + self.crop_size]
        return image, mask

    def pos_enc(self, img_name):

        info_diz = self.geo_info

        key = img_name.split("/")[1] + "-" + img_name.split("/")[2] + "-" + img_name.split("/")[-1].split(".")[0]

        x = info_diz[key]["patch_centroid_x"] - 489353.59  # center coordinate for EPSG:2154
        y = info_diz[key]["patch_centroid_y"] - 6587552.2  # center coordinate for EPSG:2154

        d = int(256 / 2)
        d_i = np.arange(0, d / 2)
        freq = 1 / (10000 ** (2 * d_i / d))
        enc = np.zeros(d * 2)
        enc[0:d:2] = np.sin(x * freq)
        enc[1:d:2] = np.cos(x * freq)
        enc[d::2] = np.sin(y * freq)
        enc[d + 1::2] = np.cos(y * freq)

        return torch.tensor(enc)

    @property
    def num_classes(self):
        return len(self.label_mapping)

    @property
    def labels_palette(self):
        palette = [item for index, sublist in self.label_mapping.items() if index != self.ignore_index for item in sublist]
        return tuple(palette)

    def __getitem__(self, i):

        # read data
        image = tiff.imread(os.path.join(self.data_dir, self.images_fps[i]))
        mask = tiff.imread(os.path.join(self.data_dir, self.masks_fps[i]))

        # select only bands of interest
        image = self.load_bands(image)

        mask[mask == 19] = 0
        mask[mask == 18] = 0
        mask[mask == 17] = 0
        mask[mask == 16] = 0
        mask[mask == 15] = 0
        mask[mask == 14] = 0
        mask[mask == 13] = 0

        # random crop of the image and the mask
        if self.crop_size:
            image, mask = self.random_crop(image, mask)

        # apply augmentations
        if self.transform:
            pair = self.transform(image=image, mask=mask)
            image, mask = pair['image'], pair['mask']

        if self.geo_info:
            coords = torch.unsqueeze(self.pos_enc(self.images_fps[i]), -1)
            coords = torch.unsqueeze(coords.expand(256, 256), 0).float()
            image = torch.cat((image, coords), dim=0)

        # return self.images_fps[i], image, mask

        if self.include_names:
            return image, mask, os.path.basename(self.images_fps[i]), os.path.basename(self.masks_fps[i])
        else:
            return image, mask

    def __len__(self):
        return len(self.images_fps)


if __name__ == "__main__":

    root_dir = sys.argv[1]

    dataset = FLAIRDataset(root_dir, stage='train', bands='rgb', crop_size=256)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

    for batch_idx, batch in enumerate(dataloader):
        images, labels = batch
        print(f"batch - {batch_idx} - ", images.shape, images.min(), images.max(), labels.shape)
