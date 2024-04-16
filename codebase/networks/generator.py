import torch
from torch import nn


class ColorGenerator(nn.Module):

    def __init__(self):
        super(ColorGenerator, self).__init__()

        w = torch.ones([256 * 256 * 256, 3], requires_grad=True)
        b = torch.zeros([256 * 256 * 256, 3], requires_grad=True)

        self.w = nn.Parameter(w)
        self.b = nn.Parameter(b)

        self.register_parameter("weight_trans", self.w)
        self.register_parameter("bias_trans", self.b)

    def forward(self, img):
        img = torch.transpose(img, 1, 3)
        img = torch.transpose(img, 1, 2)

        idx = img[:, :, :, 0] * 256 * 256 + img[:, :, :, 1] * 256 + img[:, :, :, 2]

        idx = idx.long()

        img = (img / 127.5) - 1

        img_trans = self.w[idx] * img + self.b[idx]

        img_trans = (img_trans + 1) * 127.5

        img_trans = torch.transpose(img_trans, 1, 3)
        img_trans = torch.transpose(img_trans, 2, 3)

        return img_trans
