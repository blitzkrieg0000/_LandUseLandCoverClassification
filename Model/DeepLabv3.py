import torch
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepLabv3(torch.nn.Module):
    def __init__(self, input_channels=12, segmentation_classes=9, freeze_backbone=False):
        super(DeepLabv3, self).__init__()
        self.model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)

        for param in self.model.parameters():
            param.requires_grad = freeze_backbone

        self.model.backbone.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size=(3, 3), stride=1, padding=(3, 3), bias=False)
        self.model.classifier[4] = torch.nn.Conv2d(256, segmentation_classes, kernel_size=(1, 1), stride=(1, 1))
        self.model.aux_classifier = None
    

    def forward(self, x):
        return self.model(x)
    



if __name__ == "__main__":
    model = DeepLabv3(input_channels=12, segmentation_classes=9, freeze_backbone=False)
    model = model.to(DEVICE)
    model.eval()

    print(model)

    # Dummy input
    input_image = torch.rand(8, 12, 64, 64).to(DEVICE)

    # Modeli çalıştırma
    with torch.no_grad():
        output = model(input_image)

    print("Model Output:", output["out"].shape)
