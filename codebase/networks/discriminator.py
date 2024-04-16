from torch import nn


class Discriminator(nn.Module):

    def __init__(self, num_channels, num_features=64):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=num_features, out_channels=num_features * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=num_features * 2, out_channels=num_features * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=num_features * 4, out_channels=num_features * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=num_features * 8, out_channels=1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        out = self.model(x)
        return out
