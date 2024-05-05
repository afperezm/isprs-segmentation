import pytorch_lightning as pl
import torch
import torchvision

from codebase.networks import ColorMapGenerator, PatchDiscriminator


class ColorMapGAN(pl.LightningModule):
    def __init__(self, num_dis_feats=32, num_dis_layers=2, lr_gen=0.0002, lr_dis=0.0002):
        super(ColorMapGAN, self).__init__()

        self.save_hyperparameters()
        self.automatic_optimization = False

        # Networks
        self.generator = ColorMapGenerator()
        self.discriminator = PatchDiscriminator(num_channels=3, num_features=num_dis_feats,
                                                num_layers=num_dis_layers, use_instance_norm=False)

        self.mse_loss = torch.nn.MSELoss()

    def training_step(self, batch):
        source_images, target_images = batch

        optimizer_g, optimizer_d = self.optimizers()

        # Train generator
        self.toggle_optimizer(optimizer_g)
        fake_source_images = self.generator(target_images)
        pred_fake_source_images = self.discriminator(fake_source_images)
        g_loss = self.mse_loss(pred_fake_source_images, torch.ones_like(pred_fake_source_images))
        optimizer_g.zero_grad()
        self.manual_backward(g_loss)
        optimizer_g.step()
        self.untoggle_optimizer(optimizer_g)

        # Train discriminator
        self.toggle_optimizer(optimizer_d)
        # how well can it label as real?
        pred_real_source_images = self.discriminator(source_images)
        real_loss = self.mse_loss(pred_real_source_images, torch.ones_like(pred_real_source_images))
        # how well can it label as fake?
        pred_fake_source_images = self.discriminator(fake_source_images.detach())
        fake_loss = self.mse_loss(pred_fake_source_images, torch.zeros_like(pred_fake_source_images))
        d_loss = 0.5 * (real_loss + fake_loss)
        optimizer_d.zero_grad()
        self.manual_backward(d_loss)
        optimizer_d.step()
        self.untoggle_optimizer(optimizer_d)

        self.log_dict({"train/g_loss": g_loss, "train/d_loss": d_loss}, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        source_images, target_images = batch

        current_epoch = self.current_epoch
        global_step = self.global_step
        tensorboard = self.logger.experiment

        print(f'current_epoch={current_epoch} global_step={global_step}')

        target_images_adapted = self.generator(target_images)

        grid = torchvision.utils.make_grid(source_images, normalize=True, value_range=(-1, 1))
        tensorboard.add_image(tag="valid/source_images", img_tensor=grid, global_step=current_epoch)

        grid = torchvision.utils.make_grid(target_images, normalize=True, value_range=(-1, 1))
        tensorboard.add_image(tag="valid/target_images", img_tensor=grid, global_step=current_epoch)

        grid = torchvision.utils.make_grid(target_images_adapted, normalize=True, value_range=(-1, 1))
        tensorboard.add_image(tag="valid/target_images_adapted", img_tensor=grid, global_step=current_epoch)

    def configure_optimizers(self):
        lr_gen = self.hparams.lr_gen
        lr_dis = self.hparams.lr_dis

        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=lr_gen)
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr_dis)

        return [optimizer_g, optimizer_d], []
