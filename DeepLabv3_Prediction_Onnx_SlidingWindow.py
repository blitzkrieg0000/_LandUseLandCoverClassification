from functools import reduce
import onnxruntime as ort
import numpy as np


import matplotlib.pyplot as plt
from rastervision.core.data import (RasterioSource, MultiRasterSource)
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_path = "./Weight/DeepLabv3/deeplabv3_v1_10_1800_18.08.2024_14.17.00.onnx"

def FindPrimarySource(bands):
    """
        MultiRasterSource'un birden fazla bandı stack'lerken kullanacağı referans band'ın index numarasını arar.
        En büyük shape'e sahip bandın index numarasını döndürür.
    """
    reference_band_index=0
    band_size=0
    for band_index, band in enumerate(bands):
        size = reduce(lambda x, y: x * y, band.shape[:-1])
        if size >= band_size:
            band_size = size
            reference_band_index = band_index

    return reference_band_index

# Load Model
session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


# Load Data
band_list = [
    "data/dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/data/Base/raster/L1C_2023_12_01_B04.tif",
    "data/dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/data/Base/raster/L1C_2023_12_01_B03.tif",
    "data/dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/data/Base/raster/L1C_2023_12_01_B02.tif",
    "data/dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/data/Base/raster/L1C_2023_12_01_B08.tif",
    "data/dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/data/Base/raster/L1C_2023_12_01_B05.tif",
    "data/dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/data/Base/raster/L1C_2023_12_01_B06.tif",
    "data/dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/data/Base/raster/L1C_2023_12_01_B07.tif",
    "data/dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/data/Base/raster/L1C_2023_12_01_B8A.tif",
    "data/dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/data/Base/raster/L1C_2023_12_01_B11.tif",
    "data/dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/data/Base/raster/L1C_2023_12_01_B12.tif"
]

# band_list = [
#     "/home/blitzkrieg/Downloads/spring/grid1/32UND_20180505T103021_52_996058_9_080234/32UND_20180505T103021_52_996058_9_080234_10m_RGB.tif",
#     "/home/blitzkrieg/Downloads/spring/grid1/32UND_20180505T103021_52_996058_9_080234/32UND_20180505T103021_52_996058_9_080234_10m_IR.tif",
#     "/home/blitzkrieg/Downloads/spring/grid1/32UND_20180505T103021_52_996058_9_080234/32UND_20180505T103021_52_996058_9_080234_20m.tif",
# ]


bands = []
for path in band_list:
    raster = RasterioSource(
                path,
                allow_streaming=True,
                raster_transformers=[],
                channel_order=None,
                bbox=None
            )
    
    bands+=[raster]

rasterSource = MultiRasterSource(bands, primary_source_idx=FindPrimarySource(bands))


def process_window(x, y, window_size, model_session):
    # Get Chip
    chip = rasterSource[x:x+window_size, y:y+window_size, :]

    # Normalize
    chip = chip / np.iinfo(chip.dtype).max
    chip = chip.astype(np.float32)
    chip = np.expand_dims(chip, axis=0)
    chip = np.transpose(chip, (0, 3, 1, 2))

    #! Inference Model
    chip = chip / 10000.0
    result = model_session.run([output_name], {input_name: chip})
    output = result[0]
    return output, chip


window_size = 120
stride = 120
height, width = rasterSource.shape[:2]

output_segmentation = np.zeros((height, width), dtype=np.uint8)


for x in range(0, height - window_size + 1, stride):
    for y in range(0, width - window_size + 1, stride):
        output, chip = process_window(x, y, window_size, session)
        output_class = np.argmax(output, axis=1)[0]
        output_segmentation[x:x+window_size, y:y+window_size] = output_class


# Normalize input image for visualization
raster = rasterSource[:, :, 1:4]
image = (raster - raster.min()) / (raster.max() - raster.min()) * 2.5


# Plot the original image and segmentation result
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(image, cmap="viridis")
axs[0].set_title("Input Image")

axs[1].imshow(output_segmentation, cmap="viridis")
axs[1].set_title("Segmentation Result")
plt.tight_layout()
plt.show()
