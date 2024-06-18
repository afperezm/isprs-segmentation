import cv2
import numpy as np
import os
import sys

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import verify_str_arg
from torchvision.transforms import transforms


class MassDataset(Dataset):

    class_rgb_values = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, data_dir, dataset='roads', split='train', transform=None):

        self.data_dir = data_dir
        self.transform = transform

        self.dataset = verify_str_arg(dataset, 'dataset', ('roads', 'buildings'))
        self.phase = verify_str_arg(split, 'split', ('train', 'valid', 'test'))

        self.images_dir = os.path.join(self.data_dir, f'mass_{self.dataset}', self.phase, 'sat')
        self.labels_dir = os.path.join(self.data_dir, f'mass_{self.dataset}', self.phase, 'map')

        self.images_paths = sorted(os.listdir(os.path.join(self.images_dir)))
        self.labels_paths = sorted(os.listdir(os.path.join(self.labels_dir)))

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, index):

        image = cv2.imread(os.path.join(self.images_dir, self.images_paths[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(os.path.join(self.labels_dir, self.labels_paths[index]))
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        if self.transform:
            image_and_label = np.concatenate([image, label], axis=2)
            image_and_label = self.transform(image_and_label)
            image, label = image_and_label[0:3], image_and_label[3:]

        return image, label


if __name__ == "__main__":

    root_dir = sys.argv[1]

    dataset = MassDataset(root_dir, dataset='roads', split='test', transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

    for batch_idx, batch in enumerate(dataloader):
        images, labels = batch
        print(f"batch - {batch_idx} - ", images.shape, labels.shape)
