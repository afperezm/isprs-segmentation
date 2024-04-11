import numpy as np
import torch


class ToTensor(object):
    """Convert numpy arrays to tensors."""

    def __call__(self, batch):
        source_image, target_image = batch

        # Swap axis to place number of channels in front
        source_image, target_image = np.transpose(source_image, (2, 0, 1)), np.transpose(target_image, (2, 0, 1))

        # Convert images to tensors
        source_tensor, target_tensor = torch.from_numpy(source_image), torch.from_numpy(target_image)

        return source_tensor, target_tensor
