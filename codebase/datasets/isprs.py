import cv2
import numpy as np
import os
import sys
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


class ISPRSDataset(Dataset):

    phase_train = 'train'
    phase_test = 'test'

    ignore_index = 6

    label_mapping = {
        0: [1, 1, 1],  # Impervious surfaces
        1: [0, 0, 1],  # Building
        2: [0, 1, 1],  # Low vegetation
        3: [0, 1, 0],  # Tree
        4: [1, 1, 0],  # Car
        5: [1, 0, 0],  # Clutter Background
        6: [0, 0, 0],  # No Data
    }

    def __init__(self, data_dir, is_train=True, include_names=False, transform=None):
        super(ISPRSDataset).__init__()

        self.data_dir = data_dir
        self.is_train = is_train
        self.include_names = include_names
        self.transform = transform

        if self.is_train:
            self.phase = self.phase_train
        else:
            self.phase = self.phase_test

        self.images_list = sorted(os.listdir(os.path.join(data_dir, self.phase, 'images')))
        self.labels_list = sorted(os.listdir(os.path.join(data_dir, self.phase, 'labels')))

        assert len(self.images_list) == len(self.labels_list)

    def __len__(self):
        return len(self.images_list)

    @property
    def num_classes(self):
        return len(self.label_mapping)

    @property
    def labels_palette(self):
        palette = [item for index, sublist in self.label_mapping.items() if index != self.ignore_index for item in sublist]
        palette = [255 if item == 1 else item for item in palette]
        return tuple(palette)

    def encode_label(self, label_tensor):
        label_encoded = torch.zeros((label_tensor.shape[1], label_tensor.shape[2]), dtype=torch.uint8)
        for index, color in self.label_mapping.items():
            label_encoded[(label_tensor == torch.tensor(color).unsqueeze(dim=1).unsqueeze(dim=1)).all(dim=0)] = index
        label_encoded = label_encoded.unsqueeze(dim=0)
        return label_encoded

    def __getitem__(self, index):

        image_path = os.path.join(self.data_dir, self.phase, 'images', self.images_list[index])
        label_path = os.path.join(self.data_dir, self.phase, 'labels', self.labels_list[index])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        if self.transform:
            image_and_label = np.concatenate([image, label], axis=2)
            image_and_label = self.transform(image_and_label)
            image, label = image_and_label[0:3], image_and_label[3:]

        # Encode labels assuming range was converted from 0..255 to 0..1
        label = self.encode_label(label)

        if self.include_names:
            return image, label, self.images_list[index], self.labels_list[index]
        else:
            return image, label


if __name__ == "__main__":

    root_dir = sys.argv[1]

    dataset = ISPRSDataset(data_dir=root_dir, is_train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406, 0, 0, 0], [0.229, 0.224, 0.225, 1, 1, 1])
        ]))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=1)

    for batch_idx, batch in enumerate(dataloader):
        images, labels = batch

        print(f"batch - {batch_idx} - ", images.shape, labels.shape)
