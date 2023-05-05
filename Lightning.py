import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.loggers import CSVLogger
import torch.nn as nn
import torch
import torchmetrics as tm
from torch.optim import Adam, SGD, lr_scheduler
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as tfms

from src.model import HNet
import src.config as CFG
from argparse import ArgumentParser
import sys


class HCRData(pl.LightningDataModule):
    def __init__(self, TRAIN_PATH, TEST_PATH) -> None:
        super().__init__()
        self.save_hyperparameters({"train_path": TRAIN_PATH, "test_path": TEST_PATH})
        self.TRAIN_PATH = TRAIN_PATH
        self.TEST_PATH = TEST_PATH
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        # the train & test transforms
        self.transforms = {
            "train": tfms.Compose(
                [
                    tfms.PILToTensor(),
                    tfms.AutoAugment(tfms.AutoAugmentPolicy.IMAGENET),
                    tfms.ConvertImageDtype(torch.float),
                    tfms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
            "test": tfms.Compose(
                [
                    tfms.PILToTensor(),
                    tfms.ConvertImageDtype(torch.float),
                    tfms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        }

    def setup(self, stage):

        if stage == "fit":
            train_ds = ImageFolder(
                root=self.TRAIN_PATH, transform=self.transforms["train"]
            )

            # Train/val splitting
            lengths = [
                int(len(train_ds) * 0.8),
                len(train_ds) - int(len(train_ds) * 0.8),
            ]
            self.train_ds, self.val_ds = random_split(dataset=train_ds, lengths=lengths)

        if stage == "test":
            self.test_ds = ImageFolder(
                root=self.TEST_PATH, transform=self.transforms["test"]
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_ds,
            batch_size=CFG.BATCH_SIZE,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(dataset=self.val_ds, batch_size=CFG.BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_ds, batch_size=CFG.BATCH_SIZE)


class LitHCR(pl.LightningModule):
    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss()
        self.train_accuracy = tm.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = tm.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = tm.Accuracy(task="multiclass", num_classes=num_classes)

    def configure_optimizers(self):
        optimizer = SGD(model.parameters(), lr=CFG.LR)
        scheduler = lr_scheduler.CyclicLR(
            optimizer=optimizer, base_lr=1e-5, max_lr=0.1, verbose=True
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        images, labels = batch

        outputs = self.model(images)
        _, preds = torch.max(outputs, 1)

        loss = F.cross_entropy(outputs, labels)
        acc = self.train_accuracy(preds, labels)

        self.log(name="Training_accuracy", value=acc, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        outputs = self.model(images)
        _, preds = torch.max(outputs, 1)

        loss = F.cross_entropy(outputs, labels)
        acc = self.val_accuracy(preds, labels)

        self.log(name="Validation_accuracy", value=acc, prog_bar=True, logger=True)

    def test_step(self, batch, bathc_idx):
        images, labels = batch

        outputs = self.model(images)
        _, preds = torch.max(outputs, 1)

        acc = self.test_accuracy(preds, labels)

        self.log(name="Test_accuracy", value=acc, prog_bar=True, logger=True)


if __name__ == "__main__":

    TRAIN_PATH, TEST_PATH, BEST_MODEL = "", "", ""
    MODEL = None

    parser = ArgumentParser(description="Train model for Hindi Character Recognition")
    parser.add_argument(
        "--mode",
        type=str,
        help="Train or Test ?",
        default="Train",
        choices=["Train", "Test"],
    )
    parser.add_argument(
        "--epochs", type=int, help="number of epochs", default=CFG.EPOCHS
    )
    parser.add_argument("--lr", type=float, help="learning rate", default=CFG.LR)
    parser.add_argument(
        "--model_type",
        type=str,
        help="Type of model (vyanjan/digit)",
        default="vyanjan",
    )

    args = parser.parse_args()

    if len(sys.argv) > 1:
        CFG.EPOCHS = args.epochs
        CFG.LR = args.lr

    if args.model_type == "digit":
        MODEL = HNet(num_classes=10)
        TRAIN_PATH = CFG.TRAIN_DIGIT_PATH
        TEST_PATH = CFG.TEST_DIGIT_PATH
        CFG.BEST_MODEL_PATH = CFG.BEST_MODEL_DIGIT
    else:
        MODEL = HNet(num_classes=36)
        TRAIN_PATH = CFG.TRAIN_VYANJAN_PATH
        TEST_PATH = CFG.TEST_VYANJAN_PATH
        CFG.BEST_MODEL_PATH = CFG.BEST_MODEL_VYANJAN

    # Create data module
    data_module = HCRData(TRAIN_PATH=TRAIN_PATH, TEST_PATH=TEST_PATH)

    # Lit Model
    model = LitHCR(model=MODEL, num_classes=(10 if args.model_type == "digit" else 36))

    # Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=CFG.EPOCHS,
        default_root_dir=".",
        logger=CSVLogger(save_dir=".", name="Lit_HCR_logs"),
    )

    if args.mode == "Train":
        trainer.fit(model, datamodule=data_module)
    else:
        trainer.test(
            model=model,
            datamodule=data_module,
            ckpt_path=LitHCR.load_from_checkpoint(checkpoint_path=CFG.BEST_MODEL_PATH),
        )
