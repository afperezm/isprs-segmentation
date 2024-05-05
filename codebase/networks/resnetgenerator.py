from torch import nn


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
