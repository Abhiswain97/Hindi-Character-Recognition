import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from data import transforms
from model import HNet, ResNet18
from tqdm import tqdm
import config as CFG
from argparse import ArgumentParser


def test(model_type):

    model = None

    if model_type == "digit":
        test_ds = ImageFolder(root=CFG.TEST_DIGIT_PATH, transform=transforms["test"])
        model = HNet(num_classes=10)
        model.load_state_dict(torch.load(CFG.BEST_MODEL_DIGIT))
    else:
        test_ds = ImageFolder(root=CFG.TEST_VYANJAN_PATH, transform=transforms["test"])
        model = HNet(num_classes=36)
        model.load_state_dict(torch.load(CFG.BEST_MODEL_VYANJAN))

    test_dl = DataLoader(test_ds, batch_size=CFG.BATCH_SIZE)

    model.eval()

    running_corrects = 0

    with torch.no_grad():
        for images, labels in tqdm(test_dl):

            outputs = model(images)

            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels)

        print(
            f"Test Accuracy of [{model_type}] model: {round(running_corrects.item()/len(test_ds) * 100, 3)}%"
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Test model for Hindi Character Recognition")
    parser.add_argument(
        "--model_type",
        type=str,
        help="Type of model (vyanjan/digit)",
        default="vyanjan",
    )

    args = parser.parse_args()

    test(model_type=args.model_type)
