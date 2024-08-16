from __future__ import annotations

import os
import sys
os.environ["DATA_INDEX_FILE"] = "data/dataset/.index"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import torch
import wandb
from torch.utils.data import DataLoader
from torchvision.models.segmentation import (DeepLabV3_ResNet50_Weights, deeplabv3_resnet50)
from torchvision.transforms import v2 as tranformsv2

from Dataset.RVDataset import (CustomBatchSampler, SegmentationDatasetConfig,
                               SpectralSegmentationDataset, VisualizeData, custom_collate_fn)
from Tool.Base import ChangeMaskOrder
from Tool.DataStorage import GetIndexDatasetPath


# =================================================================================================================== #
#! PARAMS
# =================================================================================================================== #
EPOCHS = 35
num_channels = 13  # Multispektral kanal sayısı
num_classes = 9    # Maskedeki sınıf sayısı
classes = torch.tensor([1, 2, 4, 5, 7, 8, 9, 10, 11])
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
PATCH_SIZE = 64
_ActivateWB = True

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
    BatchDataRepeatNumber=1,
    BatchSize=BATCH_SIZE,
    DropLastBatch=True
)

dataset = SpectralSegmentationDataset(dsConfig)
customBatchSampler = CustomBatchSampler(dataset, config=dsConfig)

print("Main Process Id:", os.getpid())
DATALOADER = DataLoader(
    dataset,
    batch_sampler=customBatchSampler,
    num_workers=0,
    persistent_workers=False, 
    pin_memory=True,
    collate_fn=custom_collate_fn,
    # multiprocessing_context = torch.multiprocessing.get_context("spawn")
)

class NormalizeSentinel2Transform(object):
    def __call__(self, inputs: torch.Tensor):
        #? Sentinel-2 verilerini [0, 1] aralığına normalize etmek için 10000'e bölme işlemi yapılır
        return inputs / 10000.0


TRANSFORM_IMAGE = tranformsv2.Compose([NormalizeSentinel2Transform()])


#! VisualizeData - Test
# VisualizeData(DATALOADER)
# for i, (inputs, targets) in enumerate(DATALOADER):
#     inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
#     print("\n", "-"*10)
#     print(f"Batch: {i}", inputs.shape, targets.shape)
#     print("-"*10, "\n")


# =================================================================================================================== #
#! Wandb
# =================================================================================================================== #
def ReadWandbKeyFromFile() -> str:
    with open("./data/asset/config/wandb/.apikey", "r") as f:
        return f.read()


def InitializeWandb():
    """ Weights & Biases """
    wandb_key = ReadWandbKeyFromFile()
    wandb.login(key=wandb_key)
    return wandb.init(
        project="LULC_Project01",       # Wandb Project Name
        entity="burakhansamli0-0-0-0",  # Wandb Entity Name
        dir="./data/",
        config={
        "learning_rate": LEARNING_RATE,
        "architecture": "DeepLabv3",
        "dataset": "Cukurova_IO_LULC",
        "epochs": EPOCHS
        },
        group="Train_DeepLabv3",
        tags=["Cukurova_IO_LULC", "DeepLabv3", "Train"],
    )

if _ActivateWB:
    InitializeWandb()


# =================================================================================================================== #
#! MODEL
# =================================================================================================================== #
# TODO conv1 Kernel 7x7 yap
class DeepLabv3(torch.nn.Module):
    def __init__(self, input_channels=12, segmentation_classes=9, freeze_backbone=False):
        super(DeepLabv3, self).__init__()
        self.model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)

        for param in self.model.parameters():
            param.requires_grad = not freeze_backbone

        self.model.backbone.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size=(3, 3), stride=1, padding=(3, 3), bias=False)
        self.model.classifier[4] = torch.nn.Conv2d(256, segmentation_classes, kernel_size=(1, 1), stride=(1, 1))
        self.model.aux_classifier = None
        self.model.classifier.add_module("softmax", torch.nn.Softmax(dim=1))

    def forward(self, x):
        return self.model(x)


model = DeepLabv3(input_channels=num_channels, segmentation_classes=num_classes)
model = model.to(DEVICE)
model.train()
print(model)


# =================================================================================================================== #
#! Compile Model
# =================================================================================================================== #
def CrossEntropyloss(pred, target):
    log_prob = torch.nn.functional.log_softmax(pred, dim=1)
    summ = -(target * log_prob).sum(dim=1)
    return summ.mean()


def DiceLoss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
    return 1 - dice.mean()


def CombinedLoss(pred, target):
    return DiceLoss(pred, target) + CrossEntropyloss(pred, target)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
# Wandb Watch
# wandb.watch(MODEL, log="all")
random_number = random.randint(0, 1000)


if "__main__" == __name__:
    # =================================================================================================================== #
    #! Train
    # =================================================================================================================== #
    totalAccuracy = 0
    for step, (inputs, targets) in enumerate(DATALOADER):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        #! Reorder Mask Values to Ordered Classes
        targets = ChangeMaskOrder(targets, classes)

        #! Mask To One Hot
        # One-hot kodlamalı tensor oluştur
        # targets = targets.unsqueeze(1)
        # one_hot_mask = torch.zeros((targets.size(0), num_classes, targets.size(2), targets.size(3)), device=DEVICE)
        # one_hot_mask.scatter_(1, targets, 1) # Sınıf indekslerini one-hot kodlamalı tensor haline getir
        
        optimizer.zero_grad()
        
        inputs = TRANSFORM_IMAGE(inputs)

        # Forward Pass
        outputs = model(inputs)["out"]
        
        loss = criterion(outputs, targets)
        # loss = CombinedLoss(outputs, one_hot_mask)

        # Backward Pass
        loss.backward()
        optimizer.step()

        # Accuracy
        with torch.no_grad():
            class_indices = torch.argmax(outputs, dim=1)
            targets = targets.squeeze(1)
            accuracy = 100*((class_indices.flatten() == targets.flatten()).sum() / PATCH_SIZE**2 /inputs.size(0))
            totalAccuracy += accuracy.item()
            print(f"Epoch {step+1}/{0}, Train Loss: {loss.item()/BATCH_SIZE}, Train Accuracy: {accuracy.item()}")
            if _ActivateWB:
                try:
                    wandb.log({
                        "epoch": step,
                        "train_loss": loss.item()/BATCH_SIZE,
                        "train_accuracy": accuracy
                    })
                except Exception as e:
                    print(e)

            print(f"Step: {step+1}, Epoch Accuracy: {totalAccuracy/(step+1)}")

            if step % 150 == 0:
                torch.save(model.state_dict(), f"./Weight/deeplabv3_v1_{random_number}_{step}.pth")
                if _ActivateWB:
                    wandb.save(f"./data/wandb/weight/deeplabv3_v1_{random_number}_{step}.pth")


    torch.save(model.state_dict(), f"./Weight/deeplabv3_v1_{random_number}_final.pth")
    if _ActivateWB:
        wandb.save(f".data/wandb/weight/deeplabv3_{random_number}_final.pth")
        wandb.finish()



