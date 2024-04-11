import cv2
import os
import sys

from torch.utils.data import Dataset, DataLoader


class UnpairedDataset(Dataset):
    def __init__(self, source_dir, target_dir, transform=None):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.transform = transform

        self.images_a = os.listdir(source_dir)
        self.images_b = os.listdir(target_dir)

        self.num_images_a = len(self.images_a)
        self.num_images_b = len(self.images_b)

        self.num_images = max(self.num_images_a, self.num_images_b)

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        path_source = os.path.join(self.source_dir, self.images_a[index % self.num_images_a])
        path_target = os.path.join(self.target_dir, self.images_b[index % self.num_images_b])

        img_source = cv2.imread(path_source)
        img_target = cv2.imread(path_target)

        if self.transform:
            img_source, img_target = self.transform((img_source, img_target))

        return img_source, img_target


if __name__ == "__main__":

    source_data_dir = sys.argv[1]
    target_data_dir = sys.argv[1]

    train_dataset = UnpairedDataset(source_dir=source_data_dir, target_dir=target_data_dir)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8)

    for batch_idx, batch in enumerate(train_dataloader):
        images_a, images_b = batch
        print(f"batch - {batch_idx} - ", images_a.shape, images_b.shape)
