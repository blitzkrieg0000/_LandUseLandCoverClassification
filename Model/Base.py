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
    