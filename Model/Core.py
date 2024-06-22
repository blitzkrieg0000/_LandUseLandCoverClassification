from __future__ import annotations
from enum import Enum
from typing import Annotated

import torch
import torch.nn as nn

from DataProcess.Dataset import DEVICE
from Model.Base import BaseModel
from Model.Resnet50 import CustomResNet50
from Model.Unet import CustomUnet
from Model.Unet3D import UNet3D

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelType(Enum):
    UNET_2D = "UNET_2D"
    UNET_3D = "UNET_3D"
    RESNET_50 = "RESNET_50"

MODEL_DATA = {
    ModelType.UNET_2D : CustomUnet,
    ModelType.UNET_3D : UNet3D,
    ModelType.RESNET_50 : CustomResNet50
}


class ModelManager():
    def __init__(self): 
        ...


    @staticmethod
    def Create(model: ModelType, **args) -> Annotated[tuple[BaseModel, torch.nn.modules.loss._Loss, torch.optim.Optimizer], "return: (BaseModel, Criterion, Optimizer)"]:
        try:
            _Model: BaseModel = MODEL_DATA[model](**args).to(DEVICE).train()
            return (
                _Model,
                *_Model.CompileModel()
            ) 
        except Exception as e:
            print(f"Model Oluşturma Hatası: {e}")


class TrainManager():
    ...


if "__main__" == __name__:
    # Create Model
    num_channels = 13  # Multispektral kanal sayısı
    num_classes = 9   # Maskedeki sınıf sayısı
