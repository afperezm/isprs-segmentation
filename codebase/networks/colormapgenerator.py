import torch
from torch import nn


class ColorMapGenerator(nn.Module):

    def __init__(self):
        super(ColorMapGenerator, self).__init__()

        self.weight = nn.Parameter(torch.empty((256 * 256 * 256, 3)))
        self.bias = nn.Parameter(torch.empty((256 * 256 * 256, 3)))

        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, img):
        img = torch.transpose(img, 1, 3)
        img = torch.transpose(img, 1, 2)

        img_r = (img[:, :, :, 0] + 1) * 127.5
        img_g = (img[:, :, :, 1] + 1) * 127.5
        img_b = (img[:, :, :, 2] + 1) * 127.5

        idx = img_r * 256 * 256 + img_g * 256 + img_b
        idx = idx.long()

        img_trans = self.weight[idx] * img + self.bias[idx]
        img_trans = torch.tanh(img_trans)

        img_trans = torch.transpose(img_trans, 1, 3)
        img_trans = torch.transpose(img_trans, 2, 3)

        return img_trans
