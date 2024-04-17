import torch
from torch import nn


class ColorGANGenerator(nn.Module):

    def __init__(self):
        super(ColorGANGenerator, self).__init__()

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

        img_trans = torch.clamp(img_trans, -1.0, 1.0)

        img_trans = (img_trans + 1) * 127.5

        img_trans = torch.transpose(img_trans, 1, 3)
        img_trans = torch.transpose(img_trans, 2, 3)

        return img_trans


class ConvBlock(nn.Module):

    def __init__(self, channels_in, channels_out):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ResNetBlock(nn.Module):

    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return self.block(x) + x


class ConvTransposeBlock(nn.Module):

    def __init__(self, channels_in, channels_out):
        super(ConvTransposeBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(channels_in, channels_out, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ResNetGenerator(nn.Module):

    def __init__(self):
        super(ResNetGenerator, self).__init__()

        self.model = nn.Sequential(
            # Encoding - First block uses reflection padding and instance norm
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            # Transformation - Nine residual blocks
            *[ResNetBlock(256) for _ in range(9)],
            # Decoding - Last block uses reflection padding but no normalization and tanh
            ConvTransposeBlock(256, 128),
            ConvTransposeBlock(128, 64),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)
