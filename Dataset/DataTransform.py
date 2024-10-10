import torch
from torchvision.transforms import v2 as transformsv2

from Model.Enum import ModelType
from Tool.Core import ChangeMaskOrder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =================================================================================================================== #
#! Data Transforms
# =================================================================================================================== #

class _3DCNN:
    class NormalizeSentinel2Transform(object):
        def __call__(self, inputs: torch.Tensor):
            #? Sentinel-2 verilerini [0, 1] aralığına normalize etmek için 10000'e bölme işlemi yapılır
            inputs = inputs[:, 0:10, :, :]  #TODO Değiştir.
            inputs = inputs.unsqueeze(1)
            return inputs / 10000.0


    class Target2OneHot(object):
        def __init__(self):
            self.__classes = torch.tensor([1, 2, 4, 5, 7, 8, 9, 10, 11])
            self.__number_of_classes = len(self.__classes)

        def __call__(self, targets):
            targets = ChangeMaskOrder(targets, self.__classes)

            #! Mask To One Hot
            targets = targets.long() # Maskeyi long yap

            # One-hot kodlamalı tensor oluştur
            one_hot_mask = torch.zeros((targets.size(0), self.__number_of_classes, targets.size(2), targets.size(3)), device=DEVICE)

            # Sınıf indekslerini one-hot kodlamalı tensor haline getir
            one_hot_mask = one_hot_mask.scatter_(1, targets, 1)
        
            return one_hot_mask.unsqueeze(1)


    class Output2Class(object):
        def __call__(self, outputs):
            class_indices = torch.argmax(outputs, dim=2)  #! Unet3D dim2
            class_indices = class_indices.squeeze(1)
            return class_indices


DATA_TRANSFORMS_BY_MODEL = {
    ModelType.UNET_3D: {
        "input_transform": transformsv2.Compose([_3DCNN.NormalizeSentinel2Transform()]),
        "output_transform": transformsv2.Compose([_3DCNN.Output2Class()]),
        "target_transform": transformsv2.Compose([_3DCNN.Target2OneHot()])
    }
}