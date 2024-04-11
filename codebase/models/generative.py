import torch
import torchvision

from codebase.networks.discriminator import Discriminator
from codebase.networks.generator import Generator
from pytorch_lightning import LightningModule


class ColorMapGAN(LightningModule):
    def __init__(self,
                 num_classes,
                 lr_gen: float = 0.0002,
                 lr_dis: float = 0.0002,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # networks
        self.generator = Generator()
        self.discriminator = Discriminator(num_channels=num_classes, num_features=64)

        self.adversarial_loss = torch.nn.MSELoss()

    def training_step(self, batch):
        source_images, target_images = batch

        optimizer_g, optimizer_d = self.optimizers()

        # Train generator
        self.toggle_optimizer(optimizer_g)

        # Generate images
        fake_source_images = self.generator(target_images)

        # Log images
        if self.global_step % 100 == 0:
            grid = torchvision.utils.make_grid(source_images, normalize=True, value_range=(0, 255))
            self.logger.experiment.add_image(tag="train/source_images", img_tensor=grid, global_step=0)

            grid = torchvision.utils.make_grid(target_images, normalize=True, value_range=(0, 255))
            self.logger.experiment.add_image(tag="train/target_images", img_tensor=grid, global_step=0)

            grid = torchvision.utils.make_grid(fake_source_images, normalize=True, value_range=(0, 255))
            self.logger.experiment.add_image(tag="train/fake_source_images", img_tensor=grid, global_step=0)

        valid = torch.ones(source_images.size(0), 1, 8, 8).type_as(source_images)
        g_loss = self.adversarial_loss(self.discriminator(fake_source_images), valid)

        self.log("train/g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # Train discriminator
        self.toggle_optimizer(optimizer_d)

        # Generate images
        fake_source_images = self.generator(target_images)

        # how well can it label as real?
        valid = torch.ones(source_images.size(0), 1, 8, 8).type_as(source_images)
        real_loss = self.adversarial_loss(self.discriminator(source_images), valid)
        # how well can it label as fake?
        fake = torch.zeros(source_images.size(0), 1, 8, 8).type_as(source_images)
        fake_loss = self.adversarial_loss(self.discriminator(fake_source_images.detach()), fake)

        d_loss = (real_loss + fake_loss) / 2

        self.log("train/d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def configure_optimizers(self):
        lr_gen = self.hparams.lr_gen
        lr_dis = self.hparams.lr_dis

        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=lr_gen)
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr_dis)

        return [optimizer_g, optimizer_d], []
