import pytorch_lightning as pl
import torchvision

from torch import nn, optim
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights


class DeepLabV3(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=0.00005, weight_decay=0.0):
        super(DeepLabV3, self).__init__()

        self.save_hyperparameters(logger=False)

        self.model = torchvision.models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        self.model.aux_classifier = None
        self.model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.metric1 = MulticlassJaccardIndex(num_classes)
        self.metric2 = MulticlassPrecision(num_classes)
        self.metric3 = MulticlassRecall(num_classes)
        self.metric4 = MulticlassF1Score(num_classes)

    def shared_step(self, batch):
        images, masks = batch

        outputs = self.model(images)['out']
        masks = masks.squeeze(dim=1).long()

        loss = self.criterion(outputs, masks)
        metric_iou = self.metric1(outputs, masks)
        metric_precision = self.metric2(outputs, masks)
        metric_recall = self.metric3(outputs, masks)
        metric_accuracy = self.metric4(outputs, masks)

        metrics = {"iou": metric_iou, "precision": metric_precision, "recall": metric_recall,
                   "accuracy": metric_accuracy}

        return loss, metrics

    def training_step(self, batch):
        loss, metrics = self.shared_step(batch)

        self.log("train/loss", loss, on_step=False, on_epoch=True)

        for metric_key, metric_value in metrics.items():
            self.log(f"train/{metric_key}", metric_value, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch):
        loss, metrics = self.shared_step(batch)

        self.log("valid/loss", loss, on_step=False, on_epoch=True)

        for metric_key, metric_value in metrics.items():
            self.log(f"valid/{metric_key}", metric_value, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, metrics = self.shared_step(batch)

        self.log("test/loss", loss, on_step=False, on_epoch=True)

        for metric_key, metric_value in metrics.items():
            self.log(f"test/{metric_key}", metric_value, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        learning_rate = self.hparams.learning_rate
        weight_decay = self.hparams.weight_decay

        parameters = [params for params in self.model.parameters() if params.requires_grad]

        optimizer = optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=7,
                                                         threshold=0.0001)

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "valid/loss"}}
