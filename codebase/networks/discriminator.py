from torch import nn


class Discriminator(nn.Module):

    def __init__(self, num_classes, ndf=64):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.ndf = ndf
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=num_classes, out_channels=ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=ndf * 8, out_channels=1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        out = self.layers(x)
        return out
