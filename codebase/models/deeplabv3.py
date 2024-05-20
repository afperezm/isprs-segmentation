import pytorch_lightning as pl
import torch
import torchmetrics
import torchvision

from PIL import Image

from torch import nn, optim
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, DeepLabV3_ResNet101_Weights, \
    DeepLabV3_MobileNet_V3_Large_Weights


def make_decoded_grid(tensor):
    grid = torchvision.utils.make_grid(tensor)[0]

    grid_image = Image.fromarray(grid.byte().cpu().numpy())
    grid_image.putpalette((255, 255, 255, 255, 0, 0, 255, 255, 0, 0, 255, 0, 0, 255, 255, 0, 0, 255))

    grid_array = grid_image.convert("RGB")
    grid_tensor = torchvision.transforms.functional.pil_to_tensor(grid_array)

    return grid_tensor


class DeepLabV3(pl.LightningModule):
    def __init__(self, num_classes, backbone='resnet50', loss_ce_weight=1.0, loss_dice_weight=0.0,
                 backbone_learning_rate=0.00005, classifier_learning_rate=0.00005,
                 backbone_weight_decay=0.0, classifier_weight_decay=0.0, **kwargs):
        super(DeepLabV3, self).__init__()

        self.save_hyperparameters(logger=False)

        if backbone == 'mobilenet':
            self.model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
                weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
        elif backbone == 'resnet50':
            self.model = torchvision.models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        elif backbone == 'resnet101':
            self.model = torchvision.models.segmentation.deeplabv3_resnet101(
                weights=DeepLabV3_ResNet101_Weights.DEFAULT)
        else:
            raise ValueError("Invalid backbone selection")

        self.model.aux_classifier = None
        if backbone == 'mobilenet':
            self.model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(960, num_classes)
        else:
            self.model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.criterion2 = torchmetrics.Dice(num_classes=num_classes)
        self.metric1 = MulticlassJaccardIndex(num_classes)
        self.metric2 = MulticlassPrecision(num_classes)
        self.metric3 = MulticlassRecall(num_classes)
        self.metric4 = MulticlassF1Score(num_classes)

    def shared_step(self, batch):
        loss_ce_weight = self.hparams.loss_ce_weight
        loss_dice_weight = self.hparams.loss_dice_weight

        images, masks = batch

        outputs = self.model(images)['out']
        masks = masks.squeeze(dim=1).long()

        loss_ce = self.criterion(outputs, masks)
        loss_dice = 1.0 - self.criterion2(outputs, masks)

        loss = loss_ce_weight * loss_ce + loss_dice_weight * loss_dice
        metric_iou = self.metric1(outputs, masks)
        metric_precision = self.metric2(outputs, masks)
        metric_recall = self.metric3(outputs, masks)
        metric_accuracy = self.metric4(outputs, masks)

        metrics = {"iou": metric_iou, "precision": metric_precision, "recall": metric_recall,
                   "accuracy": metric_accuracy, "loss_ce": loss_ce, "loss_dice": loss_dice}

        return loss, metrics, outputs

    def training_step(self, batch):
        loss, metrics, _ = self.shared_step(batch)

        self.log("train/loss", loss, on_step=False, on_epoch=True)

        for metric_key, metric_value in metrics.items():
            self.log(f"train/{metric_key}", metric_value, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics, outputs = self.shared_step(batch)

        self.log("valid/loss", loss, on_step=False, on_epoch=True)

        for metric_key, metric_value in metrics.items():
            self.log(f"valid/{metric_key}", metric_value, on_step=False, on_epoch=True)

        if batch_idx == 0:
            images, masks, predictions = batch[0], batch[1], torch.argmax(outputs, dim=1, keepdim=True)

            current_epoch = self.current_epoch
            tensorboard = self.logger.experiment

            grid = torchvision.utils.make_grid(images, normalize=True, value_range=(-1, 1))
            tensorboard.add_image(tag="valid/images", img_tensor=grid, global_step=current_epoch)

            grid = make_decoded_grid(masks)
            tensorboard.add_image(tag="valid/masks", img_tensor=grid, global_step=current_epoch)

            grid = make_decoded_grid(predictions)
            tensorboard.add_image(tag="valid/predictions", img_tensor=grid, global_step=current_epoch)

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, metrics, _ = self.shared_step(batch)

        self.log("test/loss", loss, on_step=False, on_epoch=True)

        for metric_key, metric_value in metrics.items():
            self.log(f"test/{metric_key}", metric_value, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        backbone_learning_rate = self.hparams.backbone_learning_rate
        classifier_learning_rate = self.hparams.classifier_learning_rate
        backbone_weight_decay = self.hparams.backbone_weight_decay
        classifier_weight_decay = self.hparams.classifier_weight_decay

        scheduler_factor = self.hparams.scheduler_factor
        scheduler_patience = self.hparams.scheduler_patience
        scheduler_threshold = self.hparams.scheduler_threshold

        if self.model.aux_classifier:
            grouped_parameters = [
                {'params': self.model.backbone.parameters()},
                {'params': self.model.classifier.parameters(), 'lr': classifier_learning_rate,
                 'weight_decay': classifier_weight_decay},
                {'params': self.model.aux_classifier.parameters(), 'lr': classifier_learning_rate,
                 'weight_decay': classifier_weight_decay}
            ]
        else:
            grouped_parameters = [
                {'params': self.model.backbone.parameters()},
                {'params': self.model.classifier.parameters(), 'lr': classifier_learning_rate,
                 'weight_decay': classifier_weight_decay}
            ]

        optimizer = optim.Adam(grouped_parameters, lr=backbone_learning_rate, weight_decay=backbone_weight_decay)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor,
                                                         patience=scheduler_patience,
                                                         threshold=scheduler_threshold)

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "valid/loss"}}
