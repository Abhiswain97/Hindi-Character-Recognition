import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from model import HNet, ResNet18
import config as CFG
from tqdm.auto import tqdm
from prettytable import PrettyTable
from argparse import ArgumentParser
from copy import deepcopy
from typing import Dict
import time
import logging
import sys
from data import transforms

# check is models folder exists
(CFG.BASE_PATH / "models").mkdir(exist_ok=True)


# Set up logger
logging.basicConfig(
    filename="train.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    filemode="a",
)


best_acc = 0.0


def run_one_epoch(
    epoch: int,
    ds_sizes: Dict[str, int],
    dataloaders: Dict[str, DataLoader],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: nn.Module,
    scheduler: torch.optim.lr_scheduler,
):
    """
    Run one complete train-val loop

    Parameter
    ---------

    ds_sizes: Dictionary containing dataset sizes
    dataloaders: Dictionary containing dataloaders
    model: The model
    optimizer: The optimizer
    loss: The loss

    Returns
    -------

    metrics: Dictionary containing Train(loss/accuracy) &
             Validation(loss/accuracy)

    """
    global best_acc

    metrics = {}

    for phase in ["train", "val"]:
        logging.info(f"{phase.upper()} phase")

        if phase == "train":
            model.train()
        else:
            model.eval()

        avg_loss = 0
        running_corrects = 0

        for batch_idx, (images, labels) in enumerate(
            tqdm(dataloaders[phase], total=len(dataloaders[phase]))
        ):

            images = images.to(CFG.DEVICE)
            labels = labels.to(CFG.DEVICE)

            # Zero the gradients
            optimizer.zero_grad()

            # Track history if in phase == "train"
            with torch.set_grad_enabled(phase == "train"):
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

            avg_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels)

            if batch_idx % CFG.INTERVAL == 0:
                corrects = torch.sum(preds == labels)

                logging.info(
                    f"Epoch {epoch} - {phase.upper()} - Batch {batch_idx} - Loss = {round(loss.item(), 3)} | Accuracy = {100 * corrects/CFG.BATCH_SIZE}%"
                )

        epoch_loss = avg_loss / ds_sizes[phase]
        epoch_acc = running_corrects.double() / ds_sizes[phase]

        # step the scheduler
        if phase == "train":
            scheduler.step()

        # save best model wts
        if phase == "val" and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = deepcopy(model.state_dict())
            torch.save(best_model_wts, CFG.BEST_MODEL_PATH)

        # Metrics tracking
        if phase == "train":
            metrics["train_loss"] = round(epoch_loss, 3)
            metrics["train_acc"] = round(100 * epoch_acc.item(), 3)
        else:
            metrics["val_loss"] = round(epoch_loss, 3)
            metrics["val_acc"] = round(100 * epoch_acc.item(), 3)

    return metrics


def train(dataloaders, ds_sizes, model, optimizer, criterion, scheduler):
    for epoch in range(CFG.EPOCHS):

        start = time.time()

        metrics = run_one_epoch(
            epoch=epoch,
            ds_sizes=ds_sizes,
            dataloaders=dataloaders,
            model=model,
            optimizer=optimizer,
            loss=criterion,
            scheduler=scheduler,
        )

        end = time.time() - start

        print(f"Epoch completed in: {round(end/60, 3)} mins")

        table.add_row(
            row=[
                epoch + 1,
                metrics["train_loss"],
                metrics["train_acc"],
                metrics["val_loss"],
                metrics["val_acc"],
            ]
        )
        print(table)

    # Write results to file
    with open("results.txt", "w") as f:
        results = table.get_string()
        f.write(results)


if __name__ == "__main__":

    TRAIN_PATH, TEST_PATH, BEST_MODEL = "", "", ""

    parser = ArgumentParser(description="Train model for Hindi Character Recognition")
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

    if args.model_type == "digit":
        model = HNet(num_classes=10)
        logging.info("Initialized Digit model")
        TRAIN_PATH = CFG.TRAIN_DIGIT_PATH
        CFG.BEST_MODEL_PATH = CFG.BEST_MODEL_DIGIT
    else:
        model = HNet(num_classes=36)
        logging.info("Initialized Vyanjan model")
        TRAIN_PATH = CFG.TRAIN_VYANJAN_PATH
        CFG.BEST_MODEL_PATH = CFG.BEST_MODEL_VYANJAN

    # creating the datasets
    train_ds = ImageFolder(root=TRAIN_PATH, transform=transforms["train"])

    # Train/val splitting
    lengths = [int(len(train_ds) * 0.8), len(train_ds) - int(len(train_ds) * 0.8)]
    train_ds, val_ds = random_split(dataset=train_ds, lengths=lengths)

    # creating the dataloaders
    train_dl = DataLoader(dataset=train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(dataset=val_ds, batch_size=CFG.BATCH_SIZE)

    if len(sys.argv) > 1:
        CFG.EPOCHS = args.epochs
        CFG.LR = args.lr

    # table
    table = PrettyTable(
        field_names=["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc"]
    )

    # the model
    model.to(CFG.DEVICE)

    # Setting up optimizer and loss
    optimizer = SGD(model.parameters(), lr=CFG.LR)
    criterion = CrossEntropyLoss()

    scheduler = lr_scheduler.CyclicLR(
        optimizer=optimizer, base_lr=1e-5, max_lr=0.1, verbose=True
    )

    dataloaders = {"train": train_dl, "val": val_dl}
    ds_sizes = {"train": len(train_ds), "val": len(val_ds)}

    detail = f"""
    Training details: 
    ------------------------    
    Model: {model._get_name()}
    Model Type: {args.model_type}
    Epochs: {CFG.EPOCHS}
    Optimizer: {type(optimizer).__name__}
    Loss: {criterion._get_name()}
    Learning Rate: {CFG.LR}
    Learning Rate Scheduler: {scheduler.__str__()}
    Batch Size: {CFG.BATCH_SIZE}
    Logging Interval: {CFG.INTERVAL} batches
    Train-dataset samples: {len(train_ds)}
    Validation-dataset samples: {len(val_ds)} 
    -------------------------
    """

    print(detail)

    logging.info(detail)

    start_train = time.time()

    train(
        dataloaders=dataloaders,
        ds_sizes=ds_sizes,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
    )

    end_train = time.time() - start_train

    print(f"Training completed in: {round(end_train/60, 3)} mins")
