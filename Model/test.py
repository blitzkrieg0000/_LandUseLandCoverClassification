from collections import OrderedDict

import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




#!                            3
# [1, 64, 1, 8, 8] => [1, 32, 2, 16, 16]
model =  nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1)
summary(model)

if "__main__" == __name__:


    # Örnek input
    input_tensor = torch.randn(1, 64, 1, 8, 8)
    output = model(input_tensor)
    print('Output shape:', output.shape)  

   