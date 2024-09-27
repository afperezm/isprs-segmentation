import os
import sys

import pytorch_lightning as pl
import torch

from codebase.datasets.flair import FLAIRDataset
from torch.utils.data import DataLoader


class FLAIRDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=4, num_workers=1, generator=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        if generator:
            self.generator = generator
        else:
            self.generator = torch.Generator().manual_seed(42)

        self.source_dataset = None
        self.target_dataset = None

    def setup(self, stage=None):
        self.source_dataset = FLAIRDataset(self.data_dir,
                                           os.path.join(self.data_dir, 'sub_train_imgs.txt'),
                                           os.path.join(self.data_dir, 'sub_train_masks.txt'),
                                           bands='rgb')
        self.target_dataset = FLAIRDataset(self.data_dir,
                                           os.path.join(self.data_dir, 'sub_test_imgs.txt'),
                                           os.path.join(self.data_dir, 'sub_test_masks.txt'),
                                           bands='rgb')

    def train_dataloader(self):
        return {
            'source': DataLoader(self.source_dataset, batch_size=self.batch_size, num_workers=self.num_workers // 2,
                                 shuffle=True, generator=self.generator),
            'target': DataLoader(self.target_dataset, batch_size=self.batch_size, num_workers=self.num_workers // 2,
                                 shuffle=True, generator=self.generator)
        }

    def val_dataloader(self):
        return DataLoader(self.target_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          generator=self.generator)

    def test_dataloader(self):
        return DataLoader(self.target_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          generator=self.generator)


if __name__ == "__main__":
    root_dir = sys.argv[1]

    data_module = FLAIRDataModule(root_dir, batch_size=32, num_workers=8)

    data_module.setup()

    valid_dataloader = data_module.val_dataloader()

    for i, (target_batch) in enumerate(valid_dataloader):
        target_inputs, target_labels = target_batch

        print(f"Batch {i + 1}")
        print(f"Target batch shape: {target_inputs.shape}, Labels: {target_labels.shape}")
        print("----")

    # train_loaders = data_module.train_dataloader()
    #
    # for i, (source_batch, target_batch) in enumerate(zip(train_loaders['source'], train_loaders['target'])):
    #     source_inputs, source_labels = source_batch
    #     target_inputs, target_labels = target_batch
    #
    #     print(f"Batch {i + 1}")
    #     print(f"Source batch shape: {source_inputs.shape}, Labels: {source_labels.shape}")
    #     print(f"Target batch shape: {target_inputs.shape}, Labels: {target_labels.shape}")
    #     print("----")
