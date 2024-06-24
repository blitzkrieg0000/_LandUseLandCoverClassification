from Model.Core import ModelManager
from Model.Enum import ModelType


MODEL, criterion, optimizer = ModelManager.Create(ModelType.UNET_3D)


print(MODEL)