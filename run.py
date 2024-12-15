import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

from models import *
from dataset import *

model_checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min"
)

early_stopping_callback = EarlyStopping(
    monitor="val_loss",
    patience=5,
    verbose=True,
    mode="min"
)

if __name__ == '__main__':

    max_epochs = 50
    variant = 'resnet_50'
    num_workers = 8
    batch_size = 32
    shuffle = True

    trainer = L.Trainer(max_epochs=50,
                        callbacks=[model_checkpoint_callback,
                                   early_stopping_callback])

    model = Model(ConvNeXt(variant='convnext_b',
                         pretrained=True,
                         n_classes=4))

    train_dataloader = DataLoader(supervised_abdataset_train,
                                  num_workers=8,
                                  batch_size=32,
                                  shuffle=True)

    val_dataloader   = DataLoader(supervised_abdataset_val,
                                  num_workers=8,
                                  batch_size=32,
                                  shuffle=False)

    test_dataloader  = DataLoader(supervised_abdataset_test,
                                  num_workers=8,
                                  batch_size=32,
                                  shuffle=False)

    trainer.fit(model,
                train_dataloader,
                val_dataloader)
