from dataclasses import dataclass, field
from typing import Annotated
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    MODEL_DEFAULTS = {
        "LEARNING_RATE" : 0.001
    }
    
    def __init__(self):
        super(BaseModel, self).__init__()

        
    def CompileModel(self, model: nn.Module) -> Annotated[tuple[torch.nn.modules.loss._Loss, torch.optim.Optimizer], "return: (Loss, Optimizer)"]:
        return (
           nn.BCEWithLogitsLoss(),
           torch.optim.Adam(model.parameters(), lr=BaseModel.MODEL_DEFAULTS["LEARNING_RATE"])
        )


@dataclass
class ModelMeta(object):
    NumInputChannel: int
    NumClasses: int
    PatchSize: int
    InputDepth: int = -1
    InputHeight: int = -1
    InputWidth: int = -1
    InputShape: tuple = field(init=False)
    OutputShape: tuple = field(init=False)


    def __post_init__(self):
        if self.InputHeight == -1 or self.InputWidth == -1: 
            self.InputHeight = self.InputWidth = self.PatchSize

        self.InputShape = (self.NumInputChannel, self.InputHeight, self.InputWidth)
        self.OutputShape = (self.NumClasses, self.InputHeight, self.InputWidth)