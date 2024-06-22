from typing import Annotated

import torch

from Model.Base import BaseModel
from Model.Const import MODEL_DATA
from Model.Enum import ModelType

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
