from Model.Enum import ModelType

from Model.Resnet50 import CustomResNet50
from Model.Unet import CustomUnet
from Model.Unet3D import UNet3D


MODEL_DATA = {
    ModelType.UNET_2D : CustomUnet,
    ModelType.UNET_3D : UNet3D,
    ModelType.RESNET_50 : CustomResNet50
}