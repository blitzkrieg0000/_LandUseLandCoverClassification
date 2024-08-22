import torch
from torchinfo import summary
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import (DeepLabV3_ResNet50_Weights,
                                             deeplabv3_resnet50)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CHANNELS = 10
NUM_CLASSES = 33


class DeepLabv3(torch.nn.Module):
    def __init__(self, input_channels=12, segmentation_classes=9, freeze_backbone=False):
        super(DeepLabv3, self).__init__()
        self.model = deeplabv3_resnet50(
            weights_backbone=ResNet50_Weights.IMAGENET1K_V2,
            weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        )

        for param in self.model.parameters():
            param.requires_grad = not freeze_backbone

        self.model.backbone.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=1, padding=(3, 3), bias=False)
        self.model.classifier[4] = torch.nn.Conv2d(256, segmentation_classes, kernel_size=(1, 1), stride=(1, 1))
        self.model.aux_classifier = None
        # self.model.classifier.add_module("softmax", torch.nn.Softmax(dim=1))

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = DeepLabv3(input_channels=NUM_CHANNELS, segmentation_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    model.train()

    print(model, "\n",)
    summary(model, input_size=(16, 10, 120, 120))

    # Dummy input
    input_image = torch.rand(16, 10, 120, 120).to(DEVICE)

    # Modeli çalıştırma
    with torch.no_grad():
        output = model(input_image)

    print("Model Output:", output["out"].shape)
