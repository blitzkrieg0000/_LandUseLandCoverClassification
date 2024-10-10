from __future__ import annotations

import os
import sys
os.environ["DATA_INDEX_FILE"] = "data/dataset/.index"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
from torchvision.models.segmentation import (DeepLabV3_ResNet50_Weights, deeplabv3_resnet50)
from torchvision.transforms import v2 as tranformsv2

from Dataset.RasterLoader.RVDataset import (SegmentationBatchSampler, SegmentationDatasetConfig,
                               SpectralSegmentationDataset, VisualizePrediction, CollateFN)
from Tool.Core import ChangeMaskOrder
from Tool.DataStorage import GetIndexDatasetPath


# =================================================================================================================== #
#! PARAMS
# =================================================================================================================== #
num_channels = 13  # Multispektral kanal sayısı
num_classes = 9    # Maskedeki sınıf sayısı
classes = torch.tensor([1, 2, 4, 5, 7, 8, 9, 10, 11])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
EPOCHS = 1
PATCH_SIZE = 64
old_classes = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
target_classes = torch.tensor([1, 2, 4, 5, 7, 8, 9, 10, 11])


# =================================================================================================================== #
#! DATASET
# =================================================================================================================== #
DATASET_PATH = GetIndexDatasetPath("LULC_IO_10m")

dsConfig = SegmentationDatasetConfig(
    ClassNames=["background", "excavation_area"],
    ClassColors=["lightgray", "darkred"],
    NullClass="background",
    MaxWindowsPerScene=None,                        # TODO Rasterlar arasında random ve her bir raster içinde randomu ayarla
    PatchSize=(PATCH_SIZE, PATCH_SIZE),
    PaddingSize=0,
    Shuffle=True,
    Epoch=EPOCHS,
    DatasetRootPath=DATASET_PATH,
    RandomPatch=False,
    BatchDataRepeatNumber=2,
    BatchSize=BATCH_SIZE,
    DropLastBatch=True,
    StrideSize=PATCH_SIZE
)

dataset = SpectralSegmentationDataset(dsConfig)
customBatchSampler = SegmentationBatchSampler(dataset, config=dsConfig)

print("Main Process Id:", os.getpid())
DATALOADER = DataLoader(
    dataset,
    batch_sampler=customBatchSampler,
    num_workers=0,
    persistent_workers=False, 
    pin_memory=True,
    collate_fn=CollateFN,
    # multiprocessing_context = torch.multiprocessing.get_context("spawn")
)

class NormalizeSentinel2Transform(object):
    def __call__(self, inputs: torch.Tensor):
        #? Sentinel-2 verilerini [0, 1] aralığına normalize etmek için 10000'e bölme işlemi yapılır
        return inputs / 10000.0


TRANSFORM_IMAGE = tranformsv2.Compose([NormalizeSentinel2Transform()])


# =================================================================================================================== #
#! MODEL
# =================================================================================================================== #
class DeepLabv3(torch.nn.Module):
    def __init__(self, input_channels=12, segmentation_classes=9, freeze_backbone=False):
        super(DeepLabv3, self).__init__()
        self.model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)

        for param in self.model.parameters():
            param.requires_grad = not freeze_backbone

        self.model.backbone.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=1, padding=(3, 3), bias=False)
        self.model.classifier[4] = torch.nn.Conv2d(256, segmentation_classes, kernel_size=(1, 1), stride=(1, 1))
        self.model.aux_classifier = None
        self.model.classifier.add_module("softmax", torch.nn.Softmax(dim=1))

    def forward(self, x):
        return self.model(x)
    

model = DeepLabv3(input_channels=num_channels, segmentation_classes=num_classes)
print(model)


# =================================================================================================================== #
#! Load Model
# ================================================================================================================== #
model.load_state_dict(torch.load("./Weight/deeplabv3_v1_690_5550.pth"))
model = model.to(DEVICE)
model = model.eval()

def ChangeMaskOrder2Old(mask, old_classes, target_classes):
    mapping = {oldc:newc for oldc, newc in zip(old_classes, target_classes)}
    new_mask = mask.clone()
    for oldc, newc in mapping.items():
        new_mask[mask == oldc] = newc
    return new_mask



if "__main__" == __name__:
    # =================================================================================================================== #
    #! Test
    # =================================================================================================================== #
    totalAccuracy = 0
    for step, (inputs, targets) in enumerate(DATALOADER):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        targets = ChangeMaskOrder(targets, classes)

        #! Mask To One Hot
        targets = targets.long() # Maskeyi uzun tamsayı yap
        
        # One-hot kodlamalı tensor oluştur
        targets = targets.unsqueeze(1)
        one_hot_mask = torch.zeros((targets.size(0), num_classes, targets.size(2), targets.size(3)), device=DEVICE)

        # Sınıf indekslerini one-hot kodlamalı tensor haline getir
        one_hot_mask.scatter_(1, targets, 1)

        inputs = TRANSFORM_IMAGE(inputs)
        outputs = model(inputs)["out"]
        

        # Accuracy
        with torch.no_grad():
            class_indices = torch.argmax(outputs, dim=1)
            targets = targets.squeeze(1)
            accuracy = 100*((class_indices.flatten() == targets.flatten()).sum() / PATCH_SIZE**2 /inputs.size(0))
            totalAccuracy += accuracy.item()
            print(f"Train Accuracy: {accuracy.item()}")
            print(f"Step: {step+1}, Epoch Accuracy: {totalAccuracy/(step+1)}")

        predicted = ChangeMaskOrder2Old(class_indices, old_classes, target_classes)
        VisualizePrediction(inputs, targets, predicted)