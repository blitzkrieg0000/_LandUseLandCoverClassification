from __future__ import annotations

import os
import sys
import time
os.environ["DATA_INDEX_FILE"] = "data/dataset/.index"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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
from torch.functional import F
from torch import nn
from PIL import Image
import numpy as np


# =================================================================================================================== #
#! PARAMS
# =================================================================================================================== #
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
BATCH_CHUNK_NUMBER = BATCH_SIZE
PATCH_SIZE = 120   # Window Size
STRIDE_SIZE = 64   # Sliding Window
num_channels = 10  # Multispektral kanal sayısı
num_classes = 33    # Maskedeki sınıf sayısı
classes = torch.arange(1, num_classes + 1) # torch.tensor([1, 2, 4, 5, 7, 8, 9, 10, 11]) # Maskedeki sınıflar
_ActivateWB = True


# =================================================================================================================== #
#! DATASET
# =================================================================================================================== #
# DATASET_PATH = GetIndexDatasetPath("LULC_IO_10m")
DATASET_PATH = "data/dataset/SeasoNet"
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
    DropLastBatch=True,
    StrideSize=STRIDE_SIZE,
    BatchDataChunkNumber=BATCH_CHUNK_NUMBER,
    # ChannelOrder=[1,2,3,7],
    DataFilter=[".*_10m", ".*_20m", ".*_IR"]
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
class DeepLabv3(torch.nn.Module):
    def __init__(self, input_channels=12, segmentation_classes=9, freeze_backbone=False):
        super(DeepLabv3, self).__init__()
        self.model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)

        for param in self.model.parameters():
            param.requires_grad = not freeze_backbone

        self.model.backbone.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=1, padding=(3, 3), bias=False)
        self.model.classifier[4] = torch.nn.Conv2d(256, segmentation_classes, kernel_size=(1, 1), stride=(1, 1))
        self.model.aux_classifier = None
        # self.model.classifier.add_module("softmax", torch.nn.Softmax(dim=1))

    def forward(self, x):
        return self.model(x)


model = DeepLabv3(input_channels=num_channels, segmentation_classes=num_classes)
model = model.to(DEVICE)
model.train()


##! --------------- Load Weights --------------- !##
model.load_state_dict(torch.load("Weight/deeplabv3_v1_196_500_17.08.2024_21.26.16.pth"))

print(model)


# =================================================================================================================== #
#! Compile Model
# =================================================================================================================== #
# def CrossEntropyloss(pred, target):
#     log_prob = torch.nn.functional.log_softmax(pred, dim=1)
#     summ = -(target * log_prob).sum(dim=1)
#     return summ.mean()


# def DiceLoss(pred, target, smooth=1.0):
#     pred = torch.sigmoid(pred)
#     intersection = (pred * target).sum(dim=(2, 3))
#     dice = (2.0 * intersection + smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
#     return 1 - dice.mean()


# def CombinedLoss(pred, target):
#     return DiceLoss(pred, target) + CrossEntropyloss(pred, target)


# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # pt = exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha = self.alpha[targets]
            focal_loss = alpha * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, reduction="mean"):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: BATCHxCLASSx128x128
        # targets: BATCHx128x128 (her piksel için sınıf etiketleri)

        inputs = F.softmax(inputs, dim=1)  # BATCHxCLASSx128x128
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1])  # BATCHx128x128xCLASS
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # BATCHxCLASSx128x128

        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score

        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


dice_loss_fn = DiceLoss()
# focal_loss_fn = FocalLoss(gamma=2.0)
# criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
# Wandb Watch
# wandb.watch(MODEL, log="all")

random_number = random.randint(0, 1000)



if "__main__" == __name__:


    land_cover_classes = {
        0: "Continuous urban fabric",
        1: "Discontinuous urban fabric",
        2: "Industrial or commercial units",
        3: "Road and rail networks and associated land",
        4: "Port areas",
        5: "Airports",
        6: "Mineral extraction sites",
        7: "Dump sites",
        8: "Construction sites",
        9: "Green urban areas",
        10: "Sport and leisure facilities",
        11: "Non-irrigated arable land",
        12: "Vineyards",
        13: "Fruit trees and berry plantations",
        14: "Pastures",
        15: "Broad-leaved forest",
        16: "Coniferous forest",
        17: "Mixed forest",
        18: "Natural grasslands",
        19: "Moors and heathland",
        20: "Transitional woodland/shrub",
        21: "Beaches, dunes, sands",
        22: "Bare rock",
        23: "Sparsely vegetated areas",
        24: "Inland marshes",
        25: "Peat bogs",
        26: "Salt marshes",
        27: "Intertidal flats",
        28: "Water courses",
        29: "Water bodies",
        30: "Coastal lagoons",
        31: "Estuaries",
        32: "Sea and ocean"
    }


    def WBMask(bg_img, pred_mask, true_mask):
        return wandb.Image(bg_img, masks={
            "predictions" : {"mask_data" : pred_mask, "class_labels" : land_cover_classes},
            "ground_truth" : {"mask_data" : true_mask, "class_labels" : land_cover_classes}
        })


    # =================================================================================================================== #
    #! Train
    # =================================================================================================================== #
    totalAccuracy = 0
    result_images = []
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
        
        # Loss
        loss = dice_loss_fn(outputs, targets) # + focal_loss_fn(outputs, targets)

        # Backward Pass
        loss.backward()
        optimizer.step()

        # Accuracy
        with torch.no_grad():
            class_indices = torch.argmax(outputs, dim=1)
            targets = targets.squeeze(1)
            # accuracy = 100*((class_indices.flatten() == targets.flatten()).sum() / PATCH_SIZE**2 /inputs.size(0))
            accuracy = 100 * (class_indices == targets).float().mean()

            totalAccuracy += accuracy.item()
            print(f"Epoch {step+1}/{0}, Train Loss: {loss.item()/BATCH_SIZE}, Train Accuracy: {accuracy.item()}")
            if _ActivateWB:
                try:
                    wandb.log({
                        "epoch": step,
                        "train_loss": loss.item()/BATCH_SIZE,
                        "train_accuracy": accuracy
                    })
                    inputs: torch.Tensor
                    if (1+step) % 50 == 0:
                        image:np.ndarray = inputs[0][1:4, :, :].permute(1, 2, 0).cpu().numpy()
                        image = (image - image.min()) / (image.max() - image.min())
                        wb_image = WBMask(
                            image*255,
                            class_indices[0].cpu().numpy(),
                            targets[0].cpu().numpy()
                        )
                        result_images+=[wb_image]

                except Exception as e:
                    print(e)

            print(f"Step: {step+1}, Epoch Accuracy: {totalAccuracy/(step+1)}")

            if step % 500 == 0:
                date_time_now = time.strftime("%d.%m.%Y_%H.%M.%S", time.localtime())
                torch.save(model.state_dict(), f"./Weight/deeplabv3_v1_{random_number}_{step}_{date_time_now}.pth")
                if _ActivateWB:
                    wandb.save(f"./data/wandb/weight/deeplabv3_v1_{random_number}_{step}_{date_time_now}.pth")

            if (1+step) % 100 == 0:
                wandb.log({"Segmentation Visualization": result_images})
                result_images.clear()



    torch.save(model.state_dict(), f"./Weight/deeplabv3_v1_{random_number}_final.pth")
    if _ActivateWB:
        date_time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        wandb.save(f".data/wandb/weight/deeplabv3_{random_number}_{date_time_now}_final.pth")
        wandb.finish()



