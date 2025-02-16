import torch
import torchvision.transforms as tfms

# the train & test transforms
transforms = {
    "train": tfms.Compose(
        [
            tfms.PILToTensor(),
            tfms.AutoAugment(tfms.AutoAugmentPolicy.IMAGENET),
            tfms.ConvertImageDtype(torch.float),
            tfms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "test": tfms.Compose(
        [
            tfms.PILToTensor(),
            tfms.ConvertImageDtype(torch.float),
            tfms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
}
