import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from DataProcess.Dataset import TRANSFORM_IMAGE, SentinelPatchDataset
from Dataset.DataTransform import DATA_TRANSFORMS_BY_MODEL
from model.Resnet50 import CustomResNet50
from model.Unet import CustomUnet
import wandb

from model.Unet3D import UNet3D
from model.Core import ModelGenerator, ModelType

# =================================================================================================================== #
#! PARAMS
# =================================================================================================================== #
##! --------------- Model --------------- !##
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", DEVICE)
patch_size = 64
LEARNING_RATE = 0.001
EPOCH = 50000
BATCH_SIZE = 16

##! --------------- Dataset --------------- !##
# DATA_PATH = [f"dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/data/Pansharpen/raster/PanComposite_2023-12-01.tif"]
DATA_PATH = [f"dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/data/Resample/raster/CompositeBandsDataset02_2023-12-01.tif"]
MASK_PATH = [f"dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/mask/raster/mask.tif"]
classes = torch.tensor([1, 2, 4, 5, 7, 8, 9, 10, 11])
num_classes = len(classes)


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
        "architecture": "Unet",
        "dataset": "Cukurova_IO_LULC",
        "epochs": EPOCH
        },
        group="Train_CustomUnet",
        tags=["Cukurova_IO_LULC", "CustomUnet", "Train"],
    )
# InitializeWandb()


# =================================================================================================================== #
#! Functions
# =================================================================================================================== #
def ChangeMaskOrder(mask, classes):
    mapping = {x:i for i,x in enumerate(classes)}
    new_mask = mask.clone()
    for old, new in mapping.items():
        new_mask[mask == old] = new
    return new_mask




# =================================================================================================================== #
#! Load Dataset
# =================================================================================================================== #
dataset = SentinelPatchDataset(DATA_PATH, MASK_PATH, patch_size)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


# =================================================================================================================== #
#! Compile Model
# =================================================================================================================== #
# Create Model
num_channels = 10  # Multispektral kanal sayısı
num_classes = 9    # Maskedeki sınıf sayısı
_ModelGenerator = ModelGenerator()
MODEL = _ModelGenerator.Create(ModelType.UNET_2D, num_channels=num_channels, num_classes=num_classes)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)

# Wandb Watch
# wandb.watch(MODEL, log="all")



if "__main__" == __name__:
    # =================================================================================================================== #
    #! Train
    # =================================================================================================================== #
    print(MODEL)
    for epoch in range(EPOCH):
        totalAccuracy = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            targets = ChangeMaskOrder(targets, classes)

            #! Mask To One Hot
            targets = targets.long() # Maskeyi uzun tamsayı yap

            # One-hot kodlamalı tensor oluştur
            one_hot_mask = torch.zeros((targets.size(0), num_classes, targets.size(2), targets.size(3)), device=DEVICE)

            # Sınıf indekslerini one-hot kodlamalı tensor haline getir
            one_hot_mask.scatter_(1, targets, 1)
            one_hot_mask = one_hot_mask.unsqueeze(1) #! Unet3D

            optimizer.zero_grad()
            
            inputs = inputs[:, 0:10, :, :]
            inputs = inputs.unsqueeze(1) #! Unet3D
            
            inputs = DATA_TRANSFORMS_BY_MODEL["input_transform"](inputs)
            outputs = MODEL(inputs)
            

            loss = criterion(outputs, one_hot_mask)
            loss.backward()
            optimizer.step()

            # Accuracy
            class_indices = torch.argmax(outputs, dim=2)  #! Unet3D dim2
            class_indices = class_indices.squeeze(1)
            accuracy = 100*((class_indices.flatten() == targets.flatten()).sum() / patch_size**2 /inputs.size(0))
            totalAccuracy += accuracy.item()
            print(f"Epoch {epoch+1}/{EPOCH}, Train Loss: {loss.item()/BATCH_SIZE}, Train Accuracy: {accuracy.item()}")
            try:
                wandb.log({
                    "epoch": epoch+1,
                    "train_loss": loss.item()/BATCH_SIZE,
                    "train_accuracy": accuracy
                })
            except Exception as e:
                print(e)

        # print(f"Epoch {epoch+1}/{EPOCH}, Epoch Accuracy: {totalAccuracy}")


    torch.save(MODEL.state_dict(), "./weight/custom02_unet.pth")
    # wandb.save(".data/wandb/weight/custom02_unet.pth")
    # wandb.finish()




