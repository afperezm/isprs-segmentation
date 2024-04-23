import numpy as np
import torch


class ToTensor(object):
    """Convert numpy arrays to tensors."""

    def __call__(self, batch):
        images = batch

        # Swap axis to place number of channels in front
        images = np.transpose(images, (2, 0, 1))

        # Convert images to tensors
        images_tensor = torch.from_numpy(images)

        return images_tensor
