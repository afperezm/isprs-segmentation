from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(channels_out),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class PatchGANDiscriminator(nn.Module):

    def __init__(self, num_channels, num_features=64):
        super(PatchGANDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ConvBlock(num_features, num_features * 2),
            ConvBlock(num_features * 2, num_features * 4),
            ConvBlock(num_features * 4, num_features * 8),
            nn.Conv2d(in_channels=num_features * 8, out_channels=1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        x_norm = (x.float() / 127.5) - 1
        out = self.model(x_norm)
        return out
