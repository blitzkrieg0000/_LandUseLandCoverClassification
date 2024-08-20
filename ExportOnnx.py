from __future__ import annotations

import os
import sys

from Model.DeepLabv3 import DeepLabv3

os.environ["DATA_INDEX_FILE"] = "data/dataset/.index"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import torch

# Clear GPU cache
torch.cuda.empty_cache()


# =================================================================================================================== #
#! CONSTS
# =================================================================================================================== #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =================================================================================================================== #
#! PARAMS
# =================================================================================================================== #
LULC_CLASSES = {
        0: "Continuous urban fabric",
        1: "Discontinuous urban fabric",
        2: "Industrial or commercial units",
        3: "Road and rail networks and associated land",
        4: "Port areas",
        5: "Airports",
        6: "Mineral extraction sites",
        7: "Dump sites",
        8: "Construction sites",
        9: "Green urban areas",
        10: "Sport and leisure facilities",
        11: "Non-irrigated arable land",
        12: "Vineyards",
        13: "Fruit trees and berry plantations",
        14: "Pastures",
        15: "Broad-leaved forest",
        16: "Coniferous forest",
        17: "Mixed forest",
        18: "Natural grasslands",
        19: "Moors and heathland",
        20: "Transitional woodland/shrub",
        21: "Beaches, dunes, sands",
        22: "Bare rock",
        23: "Sparsely vegetated areas",
        24: "Inland marshes",
        25: "Peat bogs",
        26: "Salt marshes",
        27: "Intertidal flats",
        28: "Water courses",
        29: "Water bodies",
        30: "Coastal lagoons",
        31: "Estuaries",
        32: "Sea and ocean"
    }
BATCH_SIZE = 16
PATCH_SIZE = 120   # Window Size
STRIDE_SIZE = 64   # Sliding Window
NUM_CHANNELS = 10  # Multispektral kanal sayısı
NUM_CLASSES = len(LULC_CLASSES)    # Maskedeki sınıf sayısı
classes = torch.arange(1, NUM_CLASSES + 1) # torch.tensor([1, 2, 4, 5, 7, 8, 9, 10, 11]) # Maskedeki sınıflar
PT_MODEL_PATH = "./Weight/DeepLabv3/deeplabv3_v1.pth"



# =================================================================================================================== #
#! CREATE MODEL
# =================================================================================================================== #
model = DeepLabv3(input_channels=NUM_CHANNELS, segmentation_classes=NUM_CLASSES)
model = model.to(DEVICE)
model.eval()

##! --------------- Load Weights --------------- !##
model.load_state_dict(torch.load(PT_MODEL_PATH))


## --------------- Show Model --------------- !##
print(model)


##! --------------- Onnx Export --------------- !##
# Get file name
modelName = os.path.basename(PT_MODEL_PATH)
dirName = os.path.dirname(PT_MODEL_PATH)
ONNX_MODEL_PATH = os.path.join(dirName, "".join(modelName.split(".")[:-1]) + ".onnx")

torch.onnx.export(
    model, 
    torch.rand(BATCH_SIZE, NUM_CHANNELS, PATCH_SIZE, PATCH_SIZE).to(DEVICE),
    ONNX_MODEL_PATH,
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    input_names = ["input"],
    output_names = ["output"],
    opset_version=11
)

print(f"ONNX modeli kaydedildi: {ONNX_MODEL_PATH}")