import torch
import torchvision
import pytorch_lightning as pl

from codebase.models.utils import ImagePool, init_weights
from codebase.networks.discriminator import PatchGANDiscriminator
from codebase.networks.generator import ColorGANGenerator, ResNetGenerator
from itertools import chain


class ColorMapGAN(pl.LightningModule):
    def __init__(self, lr_gen=0.0002, lr_dis=0.0002, log_freq=False):
        super(ColorMapGAN, self).__init__()

        self.save_hyperparameters()
        self.automatic_optimization = False

        # Networks
        self.generator = ColorGANGenerator()
        self.discriminator = PatchGANDiscriminator(num_channels=3, num_features=64)

        self.mse_loss = torch.nn.MSELoss()

    def training_step(self, batch):
        source_images, target_images = batch

        optimizer_g, optimizer_d = self.optimizers()

        fake_source_images = self.generator(target_images)

        # Train discriminator
        self.toggle_optimizer(optimizer_d)
        # # how well can it label as real?
        # valid = torch.ones(source_images.size(0), 1, 8, 8).type_as(source_images)
        pred_real_source_images = self.discriminator(source_images)
        real_loss = self.mse_loss(pred_real_source_images, torch.ones_like(pred_real_source_images))
        # # how well can it label as fake?
        # fake = torch.zeros(source_images.size(0), 1, 8, 8).type_as(source_images)
        pred_fake_source_images = self.discriminator(fake_source_images.detach())
        fake_loss = self.mse_loss(pred_fake_source_images, torch.zeros_like(pred_fake_source_images))
        d_loss = real_loss + fake_loss
        optimizer_d.zero_grad()
        self.manual_backward(d_loss)
        optimizer_d.step()
        self.untoggle_optimizer(optimizer_d)

        # Train generator
        self.toggle_optimizer(optimizer_g)
        # valid = torch.ones(source_images.size(0), 1, 8, 8).type_as(source_images)
        pred_fake_source_images = self.discriminator(fake_source_images)
        g_loss = self.mse_loss(pred_fake_source_images, torch.ones_like(pred_fake_source_images))
        optimizer_g.zero_grad()
        self.manual_backward(g_loss)
        optimizer_g.step()
        self.untoggle_optimizer(optimizer_g)

        log_freq = self.hparams.log_freq

        # Log images
        if self.global_step % log_freq == 0:
            grid = torchvision.utils.make_grid(source_images, normalize=True, value_range=(0, 255))
            self.logger.experiment.add_image(tag="train/source_images", img_tensor=grid,
                                             global_step=int(self.global_step / log_freq) % 5)

            grid = torchvision.utils.make_grid(target_images, normalize=True, value_range=(0, 255))
            self.logger.experiment.add_image(tag="train/target_images", img_tensor=grid,
                                             global_step=int(self.global_step / log_freq) % 5)

            grid = torchvision.utils.make_grid(fake_source_images, normalize=True, value_range=(0, 255))
            self.logger.experiment.add_image(tag="train/fake_source_images", img_tensor=grid,
                                             global_step=int(self.global_step / log_freq) % 5)

        self.log_dict({"train/g_loss": g_loss, "train/d_loss": d_loss}, prog_bar=True)

    def configure_optimizers(self):
        lr_gen = self.hparams.lr_gen
        lr_dis = self.hparams.lr_dis

        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=lr_gen)
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr_dis)

        return [optimizer_g, optimizer_d], []


class CycleGAN(pl.LightningModule):
    def __init__(self, lr_gen=0.0002, lr_dis=0.0002, lambda_cycle=10.0, lambda_identity=0.0):
        super(CycleGAN, self).__init__()

        self.save_hyperparameters()
        self.automatic_optimization = False

        # generator pair
        self.gen_x = ResNetGenerator()
        self.gen_y = ResNetGenerator()

        # discriminator pair
        self.dis_x = PatchGANDiscriminator(num_channels=3, num_features=64)
        self.dis_y = PatchGANDiscriminator(num_channels=3, num_features=64)

        self.fake_a = None
        self.fake_b = None

        self.fake_pool_a = ImagePool()
        self.fake_pool_b = ImagePool()

        self.criterion = torch.nn.MSELoss()

        for m in [self.gen_x, self.gen_y, self.dis_x, self.dis_y]:
            init_weights(m)

    def generator_training_step(self, img_a, img_b):
        lambda_cycle = self.hparams.lambda_cycle
        lambda_identity = self.hparams.lambda_identity

        fake_b = self.gen_x(img_a)
        cycled_a = self.gen_y(fake_b)

        fake_a = self.gen_y(img_b)
        cycled_b = self.gen_x(fake_a)

        same_b = self.gen_x(img_b)
        same_a = self.gen_y(img_a)

        # generator gen_x must fool discriminator dis_y so label is all ones (real)
        pred_fake_b = self.dis_y(fake_b)
        mse_gen_b = self.criterion(pred_fake_b, torch.ones_like(pred_fake_b))

        # generator gen_y must fool discriminator dis_x so label is all ones (real)
        pred_fake_a = self.dis_x(fake_a)
        mse_gen_a = self.criterion(pred_fake_a, torch.ones_like(pred_fake_a))

        # compute extra losses
        identity_loss = torch.nn.functional.l1_loss(same_a, img_a) + torch.nn.functional.l1_loss(same_b, img_b)

        # compute cycleLosses
        cycle_loss = torch.nn.functional.l1_loss(cycled_a, img_a) + torch.nn.functional.l1_loss(cycled_b, img_b)

        # gather all losses
        gen_loss = mse_gen_a + mse_gen_b + lambda_cycle * cycle_loss + lambda_identity * identity_loss

        # store detached generated images
        self.fake_a = fake_a.detach()
        self.fake_b = fake_b.detach()

        return gen_loss

    def discriminator_training_step(self, img_a, img_b):
        fake_a = self.fake_pool_a.query(self.fake_a)
        fake_b = self.fake_pool_b.query(self.fake_b)

        # disX checks for domain A photos
        pred_real_a = self.dis_x(img_a)
        mse_real_a = self.criterion(pred_real_a, torch.ones_like(pred_real_a))

        pred_fake_a = self.dis_x(fake_a)
        mse_fake_a = self.criterion(pred_fake_a, torch.zeros_like(pred_fake_a))

        # disY checks for domain B photos
        pred_real_b = self.dis_y(img_b)
        mse_real_b = self.criterion(pred_real_b, torch.ones_like(pred_real_b))

        pred_fake_b = self.dis_y(fake_b)
        mse_fake_b = self.criterion(pred_fake_b, torch.zeros_like(pred_fake_b))

        # gather all losses
        dis_loss = 0.5 * (mse_real_a + mse_fake_a + mse_real_b + mse_fake_b)

        return dis_loss

    def training_step(self, batch, batch_idx):

        g_optimizer, d_optimizer = self.optimizers()

        img_a, img_b = batch

        self.toggle_optimizer(g_optimizer)
        g_loss = self.generator_training_step(img_a, img_b)
        g_optimizer.zero_grad()
        self.manual_backward(g_loss)
        g_optimizer.step()
        self.untoggle_optimizer(g_optimizer)

        self.toggle_optimizer(d_optimizer)
        d_loss = self.discriminator_training_step(img_a, img_b)
        d_optimizer.zero_grad()
        self.manual_backward(d_loss)
        d_optimizer.step()
        self.untoggle_optimizer(d_optimizer)

        self.log_dict({"train/g_loss": g_loss, "train/d_loss": d_loss}, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        img_a, img_b = batch

        img_a2b = self.gen_x(img_a)
        img_b2a = self.gen_y(img_b)

        img_a2b = img_a2b * 0.5 + 0.5
        img_b2a = img_b2a * 0.5 + 0.5

        return img_a2b, img_b2a

    def configure_optimizers(self):
        lr_gen = self.hparams.lr_gen
        lr_dis = self.hparams.lr_dis

        g_optimizer = torch.optim.Adam(chain(self.gen_x.parameters(), self.gen_y.parameters()),
                                       lr=lr_gen, betas=(0.5, 0.999))

        d_optimizer = torch.optim.Adam(chain(self.dis_x.parameters(), self.dis_y.parameters()),
                                       lr=lr_dis, betas=(0.5, 0.999))

        def gamma(epoch):
            return 1 - max(0, epoch + 1 - 100) / 101

        g_scheduler = torch.optim.lr_scheduler.LambdaLR(g_optimizer, lr_lambda=gamma)
        d_scheduler = torch.optim.lr_scheduler.LambdaLR(d_optimizer, lr_lambda=gamma)

        return [g_optimizer, d_optimizer], [g_scheduler, d_scheduler]
