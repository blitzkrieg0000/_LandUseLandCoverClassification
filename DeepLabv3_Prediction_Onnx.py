from functools import reduce

import numpy as np
import onnxruntime as ort
import rasterio
import torch
from matplotlib import pyplot as plt
from rastervision.core.data import (ClassConfig, MultiRasterSource,
                                    RasterioSource, Scene,
                                    SemanticSegmentationLabelSource)
from torchvision.transforms import v2 as tranformsv2
from Model.DeepLabv3 import DeepLabv3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class NormalizeSentinel2Transform(object):
#     def __call__(self, inputs: torch.Tensor):
#         #? Sentinel-2 verilerini [0, 1] aralığına normalize etmek için 10000'e bölme işlemi yapılır
#         return inputs / 10000.0

# TRANSFORM_IMAGE = tranformsv2.Compose([NormalizeSentinel2Transform()])


def FindPrimarySource(bands):
    """
        MultiRasterSource'un birden fazla bandı stack'lerken kullanacağı referans band'ın index numarasını arar.
        En büyük shape'e sahip bandın index numarasını döndürür.
    """
    reference_band_index = 0
    band_size=0
    for band_index, band in enumerate(bands):
        size = reduce(lambda x, y: x * y, band.shape[:-1])
        if size >= band_size:
            band_size = size
            reference_band_index = band_index

    return reference_band_index


# Load Model
model_path = "Weight/DeepLabv3/deeplabv3_v1_701_16800_19.08.2024_21.52.32.onnx"

session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name



#! Load Data
# band_list = [
#     "data/dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/data/Base/raster/L1C_2023_12_01_B08.tif",
#     "data/dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/data/Base/raster/L1C_2023_12_01_B02.tif",
#     "data/dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/data/Base/raster/L1C_2023_12_01_B03.tif",
#     "data/dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/data/Base/raster/L1C_2023_12_01_B04.tif",
#     "data/dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/data/Base/raster/L1C_2023_12_01_B05.tif",
#     "data/dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/data/Base/raster/L1C_2023_12_01_B06.tif",
#     "data/dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/data/Base/raster/L1C_2023_12_01_B07.tif",
#     "data/dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/data/Base/raster/L1C_2023_12_01_B8A.tif",
#     "data/dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/data/Base/raster/L1C_2023_12_01_B11.tif",
#     "data/dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/data/Base/raster/L1C_2023_12_01_B12.tif",
# ]

band_list = [
    "/home/blitzkrieg/Downloads/spring/grid1/32UND_20180505T103021_52_996058_9_080234/32UND_20180505T103021_52_996058_9_080234_10m_RGB.tif",
    "/home/blitzkrieg/Downloads/spring/grid1/32UND_20180505T103021_52_996058_9_080234/32UND_20180505T103021_52_996058_9_080234_10m_IR.tif",
    "/home/blitzkrieg/Downloads/spring/grid1/32UND_20180505T103021_52_996058_9_080234/32UND_20180505T103021_52_996058_9_080234_20m.tif",
]

bands = []
for path in band_list:
    raster = RasterioSource(
                path,
                allow_streaming=False,
                raster_transformers=[],
                channel_order=None,
                bbox=None
            )
    
    bands+=[raster]


rasterSource = MultiRasterSource(bands, primary_source_idx=FindPrimarySource(bands), channel_order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


# Get Chip
x, y = 0, 0
chip = rasterSource[x:x+120, y:y+120, :]

# Normalize
print(np.iinfo(chip.dtype).max)
chip = chip / np.iinfo(chip.dtype).max
chip = chip.astype(np.float32)
chip = np.expand_dims(chip, axis=0)
chip = np.transpose(chip, (0, 3, 1, 2))


#! Inference Model
# chip = TRANSFORM_IMAGE(chip)
# chip = chip / 10000.0
result = session.run([output_name], {input_name: chip})
output = result[0]


output = np.argmax(output, axis=1)


#! Show Prediction
chip = np.squeeze(chip, axis=0)
image = (chip - chip.min()) / (chip.max() - chip.min())
image = np.transpose(image[1:4, :, :], (1, 2, 0))

fig, axs = plt.subplots(1, 2)
axs[0].imshow(image, cmap="viridis")
axs[1].imshow(output[0], cmap="viridis")
plt.tight_layout()
plt.show()
