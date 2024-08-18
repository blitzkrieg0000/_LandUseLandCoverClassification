from __future__ import annotations

import os
import sys
import time

from Model.DeepLabv3 import DeepLabv3
from Model.Loss import DiceLoss
from Tool.Const import RGB_COLORS

os.environ["DATA_INDEX_FILE"] = "data/dataset/.index"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random

import numpy as np
import torch
import wandb
from PIL import Image
from torch import nn
from torch.functional import F
from torch.utils.data import DataLoader
from torchvision.models.segmentation import (DeepLabV3_ResNet50_Weights,
                                             deeplabv3_resnet50)
from torchvision.transforms import v2 as tranformsv2

from Dataset.RVDataset import (CustomBatchSampler, SegmentationDatasetConfig,
                               SpectralSegmentationDataset, VisualizeData,
                               custom_collate_fn)
from Tool.Base import ChangeMaskOrder, GetTimeStampNow
from Tool.DataStorage import GetIndexDatasetPath


# Clear GPU cache
torch.cuda.empty_cache()


# =================================================================================================================== #
#! CONSTS
# =================================================================================================================== #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random_number = random.randint(0, 1000)


# =================================================================================================================== #
#! PARAMS
# =================================================================================================================== #

LULC_CLASSES = {
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

EPOCHS = 50
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
BATCH_CHUNK_NUMBER = BATCH_SIZE
PATCH_SIZE = 120   # Window Size
STRIDE_SIZE = 64   # Sliding Window
NUM_CHANNELS = 10  # Multispektral kanal sayısı
NUM_CLASSES = len(LULC_CLASSES)    # Maskedeki sınıf sayısı
classes = torch.arange(1, NUM_CLASSES + 1) # torch.tensor([1, 2, 4, 5, 7, 8, 9, 10, 11]) # Maskedeki sınıflar
_ActivateWB = True


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
#! LOAD DATA
# =================================================================================================================== #
# DATASET_PATH = GetIndexDatasetPath("LULC_IO_10m")
DATASET_PATH = "data/dataset/SeasoNet"
dsConfig = SegmentationDatasetConfig(
    ClassNames=list(LULC_CLASSES.values()),
    ClassColors=RGB_COLORS[:NUM_CLASSES],
    NullClass=None,
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
    DataFilter=[".*_10m_RGB", ".*_10m_IR", ".*_20m"],
    # ChannelOrder=[1,2,3,7]
)

dataset = SpectralSegmentationDataset(dsConfig)
customBatchSampler = CustomBatchSampler(dataset, config=dsConfig)

DATALOADER = DataLoader(
    dataset,
    batch_sampler=customBatchSampler,
    num_workers=2,
    persistent_workers=False, 
    pin_memory=True,
    collate_fn=custom_collate_fn,
    # multiprocessing_context = torch.multiprocessing.get_context("spawn")
)


##! --------------- Visualize Data --------------- !##
# VisualizeData(DATALOADER)


##! --------------- Transforms --------------- !##
class NormalizeSentinel2Transform(object):
    def __call__(self, inputs: torch.Tensor):
        #? Sentinel-2 verilerini [0, 1] aralığına normalize etmek için 10000'e bölme işlemi yapılır
        return inputs / 10000.0

TRANSFORM_IMAGE = tranformsv2.Compose([NormalizeSentinel2Transform()])


# =================================================================================================================== #
#! CREATE MODEL
# =================================================================================================================== #
model = DeepLabv3(input_channels=NUM_CHANNELS, segmentation_classes=NUM_CLASSES)
model = model.to(DEVICE)
model.train()

##! --------------- Load Weights --------------- !##
# %87 Acc => Weight/DeepLabv3/deeplabv3_v1_128_6000_18.08.2024_13.48.38.pth
# %94 Acc => Weight/DeepLabv3/deeplabv3_v1_10_1800_18.08.2024_14.17.00.pth
model.load_state_dict(torch.load("./Weight/DeepLabv3/deeplabv3_v1_19_1200_18.08.2024_17.25.56.pth"))

## --------------- Wandb Watch --------------- !##
# wandb.watch(MODEL, log="all")


## --------------- Show Model --------------- !##
print(model)

## Onnx Export
# torch.onnx.export(
#     model, 
#     torch.rand(16, 10, 120, 120).to(DEVICE),
#     "Weight/DeepLabv3/deeplabv3_v1.onnx",
#     dynamic_axes={
#         "input": {0: "batch_size"},
#         "output": {0: "batch_size"}
#     },
#     input_names = ["input"],
#     output_names = ["output"],
#     opset_version=11
# )


# =================================================================================================================== #
#! COMPILE MODEL
# =================================================================================================================== #
##! --------------- Loss --------------- !##
cross_entropy_loss_fn = torch.nn.CrossEntropyLoss()
# focal_loss_fn = FocalLoss(gamma=2.0)
dice_loss_fn = DiceLoss()


##! --------------- Optimizer --------------- !##
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)



if "__main__" == __name__:
    def WBMask(bg_img, pred_mask, true_mask):
        return wandb.Image(bg_img, masks={
            "predictions" : {"mask_data" : pred_mask, "class_labels" : LULC_CLASSES},
            "ground_truth" : {"mask_data" : true_mask, "class_labels" : LULC_CLASSES}
        })


    def ToOneHot2D(targets):
        one_hot_mask = torch.zeros((targets.size(0), NUM_CLASSES, targets.size(2), targets.size(3)), device=DEVICE)
        one_hot_mask.scatter_(1, targets, 1) # Sınıf indekslerini one-hot kodlamalı tensor haline getir
        return one_hot_mask


    # =================================================================================================================== #
    #! Train
    # =================================================================================================================== #
    totalAccuracy = 0
    result_images = []
    for step, (inputs, targets) in enumerate(DATALOADER):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        
        ##! --------------- Preprocess --------------- !##
        # Reorder Mask Values to Ordered Classes
        targets = ChangeMaskOrder(targets, classes)

        # Mask To One Hot
        # One-hot kodlamalı tensor oluştur
        # targets = targets.unsqueeze(1)
        # targets = ToOneHot2D(targets)
        
        inputs = TRANSFORM_IMAGE(inputs)

        # Zeroing The Gradients
        optimizer.zero_grad()


        ##! --------------- Forward Pass --------------- !##
        outputs = model(inputs)["out"]
        
        # Loss
        loss = cross_entropy_loss_fn(outputs, targets) + dice_loss_fn(outputs, targets) # + focal_loss_fn(outputs, targets)

        # Backward Pass
        loss.backward()

        # Optimize Weights
        optimizer.step()


        ##! --------------- Evaluate --------------- !##
        with torch.no_grad():
            class_indices = torch.argmax(outputs, dim=1)
            targets = targets.squeeze(1)
            
            # Accuracy
            accuracy = 100 * (class_indices == targets).float().mean()
            # accuracy = 100*((class_indices.flatten() == targets.flatten()).sum() / PATCH_SIZE**2 /inputs.size(0))   # Pixel Accuracy
            totalAccuracy += accuracy.item()
            print(f"Epoch {step+1}, Train Loss: {loss.item()/BATCH_SIZE}, Train Accuracy: {accuracy.item()}, Average Accuracy: {totalAccuracy/(step+1)}")

            # Wandb
            if _ActivateWB:
                try:
                    wandb.log({
                        "epoch": step,
                        "train_loss": loss.item() / BATCH_SIZE,
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

            
            # Save
            if step % 300 == 0:
                date_time_now = GetTimeStampNow()
                torch.save(model.state_dict(), f"./Weight/DeepLabv3/deeplabv3_v1_{random_number}_{step}_{date_time_now}.pth")
                if _ActivateWB:
                    wandb.save(f"./data/wandb/weight/deeplabv3_v1_{random_number}_{step}_{date_time_now}.pth")

            if (1+step) % 300 == 0:
                wandb.log({"Segmentation Visualization": result_images})
                result_images.clear()

    
    ##! --------------- Finalize --------------- !##
    torch.save(model.state_dict(), f"./Weight/DeepLabv3/deeplabv3_v1_{random_number}_final.pth")
    if _ActivateWB:
        wandb.save(f".data/wandb/weight/deeplabv3_{random_number}_{GetTimeStampNow()}_final.pth")
        wandb.finish()



