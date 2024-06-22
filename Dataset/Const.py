from Dataset.Core import SentinelPatchDataset
from Dataset.Enum import DatasetType
from Model.Resnet50 import CustomResNet50
from Model.Unet import CustomUnet
from Model.Unet3D import UNet3D


# DATA_PATH = [f"dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/data/Pansharpen/raster/PanComposite_2023-12-01.tif"]
DATA_PATH = [f"data/dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/data/Resample/raster/CompositeBandsDataset02_2023-12-01.tif"]
MASK_PATH = [f"data/dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/mask/raster/mask.tif"]


DATASET_FOR_MODEL = {
    DatasetType.Cukurova_IO_LULC: SentinelPatchDataset,
    # DatasetType.Cukurova_IO_LULC_3D: Sentinel3DPatchDataset
    
}



""" [
    ,
    UNet3D,
    CustomResNet50
]"""