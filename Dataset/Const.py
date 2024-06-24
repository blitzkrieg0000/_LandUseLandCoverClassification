from Dataset.Dataset import SentinelDatasetProcessor
from Dataset.Enum import DatasetType
from Model.Resnet50 import CustomResNet50
from Model.Unet import CustomUnet
from Model.Unet3D import UNet3D


DATASET = {
    DatasetType.Cukurova_IO_LULC: SentinelDatasetProcessor,
    DatasetType.Cukurova_IO_LULC_Pan: SentinelDatasetProcessor,
}

DATASET_CONFIG = {
    SentinelDatasetProcessor : {
        # "DATA" : [f"dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/data/Pansharpen/raster/PanComposite_2023-12-01.tif"],
        "DATA" : [f"./data/dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/data/Resample/raster/CompositeBandsDataset02_2023-12-01.tif"],
        "MASK" : [f"./data/dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/mask/raster/mask.tif"],
        "PATCH_SIZE" : 64,
        "BATCH_SIZE" : 16,
        "SHUFFLE" : False
    },
    SentinelDatasetProcessor : {
        # "DATA" : [f"dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/data/Pansharpen/raster/PanComposite_2023-12-01.tif"],
        "DATA" : [f"./data/dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/data/Resample/raster/CompositeBandsDataset02_2023-12-01.tif"],
        "MASK" : [f"./data/dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/mask/raster/mask.tif"],
        "PATCH_SIZE" : 64,
        "BATCH_SIZE" : 16,
        "SHUFFLE" : False
    }
}