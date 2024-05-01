import cv2
import numpy as np
import os
import sys

from torch.utils.data import Dataset, DataLoader


class ISPRSDataset(Dataset):

    phase_train = 'train'
    phase_test = 'test'

    label_mapping = {0: [255, 255, 255],  # Impervious surfaces
                     1: [0, 0, 255],  # Building
                     2: [0, 255, 255],  # Low vegetation
                     3: [0, 255, 0],  # Tree
                     4: [255, 255, 0],  # Car
                     5: [255, 0, 0],  # Clutter Background
                     6: [0, 0, 0]  # Boundary
                     }

    def __init__(self, data_dir, is_train=True, transform=None):
        super(ISPRSDataset).__init__()

        self.data_dir = data_dir
        self.is_train = is_train
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

    def encode_label(self, label):
        label_encoded = np.zeros(label.shape[0:2], dtype=np.uint8)
        for index, color in self.label_mapping.items():
            label_encoded[np.all(label == color, axis=-1)] = index
        return label_encoded

    def __getitem__(self, index):

        image_path = os.path.join(self.data_dir, self.phase, 'images', self.images_list[index])
        label_path = os.path.join(self.data_dir, self.phase, 'labels', self.labels_list[index])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        label = self.encode_label(label)

        if self.transform:
            image, label = self.transform(image, label)

        return image, label


if __name__ == "__main__":

    root_dir = sys.argv[1]

    dataset = ISPRSDataset(data_dir=root_dir, is_train=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    for batch_idx, batch in enumerate(dataloader):
        images, labels = batch['image'], batch['label']

        print(f"batch - {batch_idx} - ", images.shape, labels.shape)
