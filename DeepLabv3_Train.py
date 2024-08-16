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
num_channels = 13  # Multispektral kanal sayısı
num_classes = 9    # Maskedeki sınıf sayısı
classes = torch.tensor([1, 2, 4, 5, 7, 8, 9, 10, 11])
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 50
patch_size = 64


# =================================================================================================================== #
#! DATASET
# =================================================================================================================== #
DATASET_PATH = GetIndexDatasetPath("LULC_IO_10m")

dsConfig = SegmentationDatasetConfig(
    ClassNames=["background", "excavation_area"],
    ClassColors=["lightgray", "darkred"],
    NullClass="background",
    MaxWindowsPerScene=None,                        # TODO Rasterlar arasında random ve her bir raster içinde randomu ayarla
    PatchSize=(patch_size, patch_size),
    PaddingSize=0,
    Shuffle=True,
    Epoch=EPOCHS,
    DatasetRootPath=DATASET_PATH,
    RandomPatch=False,
    BatchDataRepeatNumber=2,
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


#! VisualizeData
# VisualizeData(DATALOADER)
for i, (inputs, targets) in enumerate(DATALOADER):
    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
    print("\n", "-"*10)
    print(f"Batch: {i}", inputs.shape, targets.shape)
    print("-"*10, "\n")


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
        "epochs": 100
        },
        group="Train_CustomUnet",
        tags=["Cukurova_IO_LULC", "DeepLabv3", "Train"],
    )

# InitializeWandb()


# =================================================================================================================== #
#! MODEL
# =================================================================================================================== #
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
    

model = DeepLabv3(input_channels=num_channels, segmentation_classes=num_classes)
model = model.to(DEVICE)
print(model)


# =================================================================================================================== #
#! Compile Model
# =================================================================================================================== #
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
# Wandb Watch
# wandb.watch(MODEL, log="all")



if "__main__" == __name__:
    # =================================================================================================================== #
    #! Train
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

        optimizer.zero_grad()
        
        inputs = TRANSFORM_IMAGE(inputs)
        outputs = model(inputs)["out"]
        

        loss = criterion(outputs, one_hot_mask)
        loss.backward()
        optimizer.step()

        # Accuracy
        class_indices = torch.argmax(outputs, dim=1)  #! Unet3D dim2
        targets = targets.squeeze(1)
        accuracy = 100*((class_indices.flatten() == targets.flatten()).sum() / patch_size**2 /inputs.size(0))
        totalAccuracy += accuracy.item()
        print(f"Epoch {step+1}/{0}, Train Loss: {loss.item()/BATCH_SIZE}, Train Accuracy: {accuracy.item()}")
        # try:
        #     wandb.log({
        #         "epoch": step,
        #         "train_loss": loss.item()/BATCH_SIZE,
        #         "train_accuracy": accuracy
        #     })
        # except Exception as e:
        #     print(e)

        print(f"Step: {step+1}, Epoch Accuracy: {totalAccuracy/(step+1)}")

        # if step % 100 == 0:
        #     torch.save(model.state_dict(), f"./Weight/deeplabv3_v1.{random.randint(0, 1000)}.pth")
            # wandb.save(f"./data/wandb/weight/deeplabv3_v1.{random.randint(0, 1000)}.pth")


    # torch.save(model.state_dict(), "./Weight/deeplabv3_v1_final.pth")
    # wandb.save(".data/wandb/weight/deeplabv3_final.pth")
    # wandb.finish()





