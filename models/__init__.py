import torch
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import lightning as L

from .convnext import ConvNeXt
from .resnet import ResNet
from .swin import Swin
from .stn import STN

CLASS_MAP = {
    "1"  : 0,
    "2-1": 1,
    "2-2": 2,
    "3"  : 3,
}

class Model(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.accuracy = torchmetrics.Accuracy(num_classes=len(CLASS_MAP), average="macro", task="multiclass")
        self.cm = torchmetrics.ConfusionMatrix(num_classes=len(CLASS_MAP), task="multiclass")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        image, bbox, coords, y = batch
        y_hat, z = self(bbox)

        loss = F.cross_entropy(y_hat, y)
        self.log("loss", loss)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        image, bbox, coords, y = batch
        y_hat, z = self(bbox)

        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss, on_epoch=True)

        # Update accuracy and confusion matrix with the current batch
        preds = torch.argmax(y_hat, dim=1)
        self.accuracy.update(preds, y)
        self.cm.update(preds, y)

        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        # Compute and log the accuracy and confusion matrix at the end of each epoch
        val_accuracy = self.accuracy.compute()
        cm_result = self.cm.compute()
        
        self.log("val_accuracy", val_accuracy, on_epoch=True)
        print(cm_result)

        # Reset metrics for the next epoch
        self.accuracy.reset()
        self.cm.reset()

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-5)
        return optimizer
