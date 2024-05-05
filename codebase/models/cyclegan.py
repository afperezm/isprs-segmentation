""" PyTorch CycleGAN.

Credits:
This code is partially adapted from Deepak H R's Lightning CycleGAN: https://github.com/deepakhr1999/cyclegans
"""

import pytorch_lightning as pl
import random
import torch
import torchvision

from codebase.networks import ResNetGenerator, PatchDiscriminator
from itertools import chain


class CycleGAN(pl.LightningModule):
    def __init__(self, lr_gen=0.0002, lr_dis=0.0002, lambda_cycle=10.0, lambda_identity=0.0):
        super(CycleGAN, self).__init__()

        self.save_hyperparameters()
        self.automatic_optimization = False

        # generator pair
        self.gen_x = ResNetGenerator()
        self.gen_y = ResNetGenerator()

        # discriminator pair
        self.dis_x = PatchDiscriminator(num_channels=3, num_features=64)
        self.dis_y = PatchDiscriminator(num_channels=3, num_features=64)

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

    def validation_step(self, batch):
        img_a, img_b = batch

        current_epoch = self.current_epoch
        tensorboard = self.logger.experiment

        fake_b = self.gen_x(img_a)
        fake_a = self.gen_y(img_b)

        grid = torchvision.utils.make_grid(img_a, normalize=True, value_range=(-1, 1))
        tensorboard.add_image(tag="valid/source_images", img_tensor=grid, global_step=current_epoch)

        grid = torchvision.utils.make_grid(fake_b, normalize=True, value_range=(-1, 1))
        tensorboard.add_image(tag="valid/source_images_adapted", img_tensor=grid, global_step=current_epoch)

        grid = torchvision.utils.make_grid(img_b, normalize=True, value_range=(-1, 1))
        tensorboard.add_image(tag="valid/target_images", img_tensor=grid, global_step=current_epoch)

        grid = torchvision.utils.make_grid(fake_a, normalize=True, value_range=(-1, 1))
        tensorboard.add_image(tag="valid/target_images_adapted", img_tensor=grid, global_step=current_epoch)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        img_a, img_b = batch[0], batch[1]

        img_a2b = self.gen_x(img_a)
        img_b2a = self.gen_y(img_b)

        img_a2b = img_a2b * 0.5 + 0.5
        img_b2a = img_b2a * 0.5 + 0.5

        if len(batch) == 4:
            return img_a2b, img_b2a, batch[2], batch[3]
        else:
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


class ImagePool:
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size=50):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_images = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_images < self.pool_size:  # If the buffer is not full; keep inserting current images to the buffer
                self.num_images = self.num_images + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # By 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:  # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)  # collect all the images and return
        return return_images


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
            torch.nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>
