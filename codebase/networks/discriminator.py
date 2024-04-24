from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, channels_in, channels_out, use_instance_norm=True):
        super(ConvBlock, self).__init__()

        block = (nn.Conv2d(channels_in, channels_out, kernel_size=4, stride=2, padding=1),
                 nn.InstanceNorm2d(channels_out),
                 nn.LeakyReLU(negative_slope=0.2, inplace=True))

        if not use_instance_norm:
            block = block[0], block[2]

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class PatchGANDiscriminator(nn.Module):

    def __init__(self, num_channels, num_features=64, num_layers=3, use_instance_norm=True):
        super(PatchGANDiscriminator, self).__init__()

        self.model = nn.Sequential(
            # nn.Conv2d(in_channels=num_channels, out_channels=num_features, kernel_size=4, stride=2, padding=1),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ConvBlock(num_channels, num_features, False),
            # ConvBlock(num_features, num_features * 2),
            # ConvBlock(num_features * 2, num_features * 4),
            # ConvBlock(num_features * 4, num_features * 8),
            *[ConvBlock((2 ** i) * num_features, (2 ** (i + 1)) * num_features, use_instance_norm) for i in
              range(num_layers)],
            # nn.Conv2d(in_channels=num_features * 8, out_channels=1, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(in_channels=(2 ** num_layers) * num_features, out_channels=1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.model(x)
