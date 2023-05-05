from torchvision.models import resnet18
import torch.nn as nn
import torch.nn.functional as F


def calculate_conv_output(IH, IW, KH, KW, P, S):
    return (IH - KH + 2 * P) / S + 1, (IW - KW + 2 * P) / S + 1


class HNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()

        # 32 x 32 x 3 => 28 x 28 x 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(5, 5))

        # 28 x 28 x 16 => 26 x 26 x 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3))

        # 26 x 26 x 32 => num_classes
        self.fc1 = nn.Linear(26 * 26 * 32, num_classes)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 26 * 26 * 32)
        x = self.dropout(x)
        x = self.fc1(x)

        return x


class ResNet18(nn.Module):
    def __init__(self, freeze=True, num_classes=10):
        super(ResNet18, self).__init__()
        self.resnet = resnet18(pretrained=True)

        # freeze all layers if required
        if freeze:
            self.freeze_layers()

        # new layers by default have requires_grad=True
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x

    def freeze_layers(self):
        for param in self.resnet.parameters():
            param.requires_grad = False
