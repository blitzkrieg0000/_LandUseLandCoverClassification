import os
import sys
import time
import multiprocessing as mp
sys.path.append(os.getcwd())

#! MODULES
from Dataset.RasterLoader.Core import BatchSamplerMode, GeoSegmentationDataset, GeoSegmentationDatasetBatchSampler
from Dataset.RasterLoader.Default import SegmentationDatasetConfig
from Dataset.RasterLoader.Proxy import SharedArtifacts
from Dataset.RasterLoader.Util import CollateFN
from Tool.Core import ConsoleLog, GetColorsFromPalette, LogColorDefaults
from torch.utils.data import DataLoader


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



if "__main__" == __name__:
	try:
		mp.set_start_method("spawn")
	except:
		...

	# Config 1
	DATASET_PATH = "data/dataset/SeasoNet/"
	SHARED_ARTIFACTS = SharedArtifacts()
	Config = SegmentationDatasetConfig(
		ClassNames=list(LULC_CLASSES.values()),
		ClassColors=GetColorsFromPalette(len(LULC_CLASSES)),
		NullClass=None,
		MaxWindowsPerScene=None,                         # TODO Rasterlar arasında random ve her bir raster içinde randomu ayarla
		PatchSize=(120, 120),
		PaddingSize=0,
		Shuffle=True,
		DatasetRootPath=DATASET_PATH,
		RandomLimit=0,
		RandomPatch=False,
		BatchDataChunkNumber=4,
		BatchSize=4,
		DropLastBatch=True,
		DataFilter=[".*_10m_RGB", ".*_10m_IR", ".*_20m"],
		# ChannelOrder=[1,2,3,7],
		DataLoadLimit=13,
		Verbose=True
	)

	# Config 2
	# DATASET_PATH = "data/dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2"
	# SHARED_ARTIFACTS = SharedArtifacts()
	# Config = SegmentationDatasetConfig(
	# 	ClassNames=["background", "excavation_area"],
	# 	ClassColors=["lightgray", "darkred"],
	# 	NullClass="background",
	# 	MaxWindowsPerScene=None,                         # TODO Rasterlar arasında random ve her bir raster içinde randomu ayarla
	# 	PatchSize=(224, 224),
	# 	PaddingSize=0,
	# 	Shuffle=True,
	# 	DatasetRootPath=DATASET_PATH,
	# 	RandomLimit=0,
	# 	RandomPatch=False,
	# 	BatchDataChunkNumber=4,
	# 	BatchSize=16,
	# 	DropLastBatch=False,
	# 	StrideSize=112,
	# 	# ChannelOrder=[1,2,3,7],
	# 	# DataFilter=[".*_10m_RGB", ".*_10m_IR", ".*_20m"],
	# 	# DataLoadLimit=20
	# )


	#! DATASET
	dataset = GeoSegmentationDataset(Config, SHARED_ARTIFACTS)
	customBatchSampler = GeoSegmentationDatasetBatchSampler(dataset, data_split=[0.8, 0.1, 0.1], mode=BatchSamplerMode.Train)

	#! DATALOADER
	NUM_WORKERS = 2
	DATA_LOADER = DataLoader(
		dataset,
		batch_sampler=customBatchSampler,
		num_workers=NUM_WORKERS,
		persistent_workers=NUM_WORKERS>0,
		pin_memory=True,
		collate_fn=CollateFN,
		multiprocessing_context=mp.get_context("spawn") if NUM_WORKERS>0 else None
	)
	
	ConsoleLog(f"Main Process Id: {os.getpid()}", LogColorDefaults.Warning, bold=True, underline=True, blink=True)
	
	#! SHOW RESULTS
	for epoch in range(2):
		print(f"\n------------------------{epoch}------------------------\n")
		customBatchSampler.SetMode(BatchSamplerMode.Train)
		
		for i, (inputs, targets) in enumerate(DATA_LOADER):
			ConsoleLog(f"Step: {i} {inputs.shape} {targets.shape}", LogColorDefaults.Success)

		print("\n===\n")	
		customBatchSampler.SetMode(BatchSamplerMode.Test)
		for i, (inputs, targets) in enumerate(DATA_LOADER):
			ConsoleLog(f"Step: {i} {inputs.shape} {targets.shape}", LogColorDefaults.Success)

		print("\n------------------------------------------------\n")
		time.sleep(3)

	#! VisualizeData
	# VisualizeData(TRAIN_LOADER)
	
