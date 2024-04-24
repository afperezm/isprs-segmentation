import cv2
import os
import sys

from torch.utils.data import Dataset, DataLoader


class UnpairedDataset(Dataset):

    train_phase = 'train'
    test_phase = 'test'

    def __init__(self, source_dir, target_dir, is_train=True, transform=None):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.is_train = is_train
        self.transform = transform

        if self.is_train:
            self.phase = self.train_phase
        else:
            self.phase = self.test_phase

        self.images_a = os.listdir(os.path.join(source_dir, self.phase, 'images'))
        self.images_b = os.listdir(os.path.join(target_dir, self.phase, 'images'))

        self.num_images_a = len(self.images_a)
        self.num_images_b = len(self.images_b)

        self.num_images = max(self.num_images_a, self.num_images_b)

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        path_source = os.path.join(self.source_dir, self.phase, 'images', self.images_a[index % self.num_images_a])
        path_target = os.path.join(self.target_dir, self.phase, 'labels', self.images_b[index % self.num_images_b])

        img_source = cv2.imread(path_source)
        img_source = cv2.cvtColor(img_source, cv2.COLOR_BGR2RGB)
        img_target = cv2.imread(path_target)
        img_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB)

        if self.transform:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

        return img_source, img_target


if __name__ == "__main__":

    source_data_dir = sys.argv[1]
    target_data_dir = sys.argv[2]

    train_dataset = UnpairedDataset(source_dir=source_data_dir, target_dir=target_data_dir)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8)

    for batch_idx, batch in enumerate(train_dataloader):
        images_a, images_b = batch
        print(f"batch - {batch_idx} - ", images_a.shape, images_b.shape)
