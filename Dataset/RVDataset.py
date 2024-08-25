from __future__ import annotations

from enum import Enum

from multiprocessing import Manager
from multiprocessing.managers import DictProxy
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Dataset.FileReader import GeoDataReader
import random
from typing import Annotated, Any, List, NamedTuple, Set, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from pydantic import BaseModel
from rastervision.core.data import (ClassConfig, MultiRasterSource,
									Scene,
									SemanticSegmentationLabelSource)

from rastervision.pytorch_learner import (
	SemanticSegmentationRandomWindowGeoDataset,
	SemanticSegmentationSlidingWindowGeoDataset,
	SemanticSegmentationVisualizer)

from torch.utils.data import DataLoader, Dataset, Sampler, random_split


from Tool.Base import ChangeMaskOrder, GetColorsFromPalette, LimitedCache
from Tool.Util import (DataSourceMeta)
from multiprocessing import Manager
import seaborn as sb
random.seed(72)

# =================================================================================================================== #
#! CONSTS
# =================================================================================================================== #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# =================================================================================================================== #
#! FUNCTION
# =================================================================================================================== #
def DataChunkRepeatCounts(batch_size, batch_data_chunk_number):
	"""[1 1 1 1 1 1 1 1], [2 2 2 2], [4 4], [3 3 2], [8]"""
	chunkSize = torch.clamp(torch.tensor(batch_data_chunk_number), 1, batch_size)
	chunks = torch.chunk(torch.arange(batch_size), chunkSize) 
	return list(map(len, chunks))


def WorkerInitFN(worker_id):
	worker_info = torch.utils.data.get_worker_info()
	dataset = worker_info.dataset
	dataset.set_worker_info(worker_id, worker_info.num_workers)


def CollateFN(batch):
	# TODO Eksik olan None değerler, tekrar burada verisetinden çekilebiliyor mu? Dene (Recursive gibi olabilir).
	data, label = zip(*batch)
	return torch.stack([d for d in data if d is not None]), torch.stack([l for l in label if l is not None])


def ShowDatasetViaVisualizer(dataset):
	vis = SemanticSegmentationVisualizer(class_names=["background", "foreground"], class_colors=["black", "white"])
	x, y = vis.get_batch(dataset, 4)
	vis.plot_batch(x, y, show=True)


def ShowBatchRaster(raster):
	fig, ax = plt.subplots(4, 4, figsize=(7, 7))
	for i in range(12):
		ax[i%4, i//4].matshow(raster[:, :, [i]], cmap="viridis")
		ax[i%4, i//4].axis("off")

	plt.show()


def VisualizeData(dataloader, limit=None):
	# print("\nDataloader Size:", len(DATALOADER))
	
	fig, axs = plt.subplots(4, 4, figsize=(12, 12))

	for i, (buffer, mask) in enumerate(dataloader):
		print(i, buffer.shape, mask.shape, "\n-----------------\n" )
		for bn in range(buffer.shape[0]):
			for i in range(16):
				axs[i%4, i//4].axis("off")
				if i<buffer.shape[1]:
					axs[i%4, i//4].imshow(buffer[bn, i].numpy(), cmap="viridis")
					axs[i%4, i//4].set_title(f"Band {i+1}")
			print("Labels: ", mask[bn].unique())
			axs[3, 3].imshow(mask[bn])
			axs[3, 3].set_title("Ground Truth")
			plt.pause(1)
		
		if limit is not None and i >= limit:
			break

	plt.tight_layout()
	plt.show()


def VisualizeClassDistribution(dataloader, num_classes=9):
	fig, ax = plt.subplots()
	plt.tight_layout()

	uLabels = set()
	hist = np.zeros(num_classes)
	colors = sb.color_palette("husl", len(hist))
	classes = torch.arange(1, num_classes + 1)

	print("Main Process Id:", os.getpid())
	for i, (inputs, targets) in enumerate(dataloader):
		# inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
		print(f"Step: {i}", inputs.shape, targets.shape)

		targets = ChangeMaskOrder(targets, classes)

		uniqueLabels: torch.Tensor = targets.unique()
		uLabels.update(set(uniqueLabels.tolist()))
		for label in uniqueLabels:
			hist[label] += 1
		
		print("Labels: ", uLabels)
		
		ax.bar(np.arange(num_classes), hist, color=colors)
		ax.set_xticks(np.arange(num_classes))
		plt.pause(0.1)

	plt.show()


def VisualizePrediction(buffer, mask, predicted):
	# print("\nDataloader Size:", len(DATALOADER))
	
	fig, axs = plt.subplots(4, 4, figsize=(12, 12))

	for bn in range(buffer.shape[0]):
		for i in range(16):
			axs[i%4, i//4].axis("off")
			if i<buffer.shape[1]:
				image = buffer[bn, i].cpu().numpy()
				axs[i%4, i//4].imshow(image, cmap="viridis")
				axs[i%4, i//4].set_title(f"Band {i+1}")
		
		axs[2, 3].imshow(mask[bn].cpu().numpy(), cmap="viridis")
		axs[2, 3].set_title("Ground Truth")
		axs[3, 3].imshow(predicted[bn].cpu().numpy(), cmap="viridis")
		axs[3, 3].text(0, 0, "Predicted", fontsize=12, color="blue", weight="bold")

	plt.tight_layout()
	plt.show()



# =================================================================================================================== #
#! CLASS
# =================================================================================================================== #
class SharedSetStorage():
    def __init__(self):
        self.manager = Manager()
        self.items_set = self.manager.dict()


    def Add(self, key:str, value:Any=None):
        """Öge, Storage'da Tanımlı değilse ekle. (Sette yoksa ekle)."""
        if key not in self.items_set:
            self.items_set[key] = value or key


    def Get(self):
        """Storage'dan son ögeyi al. (Sette yoksa ekle)."""
        return self.items_set.values()


    def ToSet(self):
        """Storage'deki elemanları Set'e dönüştür."""
        return set(self.items_set.values())


    def __contains__(self, item):
        """Storage'de olup olmadığını kontrol eder."""
        return item in self.items_set


    def __len__(self):
        """Storage'daki eleman sayısını verir."""
        return len(self.items_set)


    def Remove(self, items):
        """Storage'daki elemanları kaldırır."""
        for item in items:
            if item in self.items_set:
                self.items_set.pop(item, None)


    def Empty(self):
        """Storage boş mu kontrol eder."""
        return len(self.items_set)==0


    def Clear(self):
        self.items_set.clear()



class SharedArtifacts():
	"""Shared Memory aracılığı ile processler arası paylaşılan ögelerin tutulmasını sağlayan bir sınıftır."""
	ExpiredScenes = SharedSetStorage()
	AvailableInSource: DictProxy[Any, TrackableIteratorState] = Manager().dict()



class TrackableIteratorState(NamedTuple):
	Id: int
	Index: int
	Available: int
	Expired: bool



class SegmentationDatasetConfig(BaseModel):
	"""Segmentation Veri Seti Konfigürasyon Sınıfı"""
	ClassNames: Annotated[List[str], "Class Names"]
	ClassColors: Annotated[List, "Class Colors"]
	NullClass: Annotated[str, "Null Class"] = None
	MaxWindowsPerScene: Annotated[int | float | None, "Max Windows Per Scene"]
	DatasetRootPath: Annotated[str, "Dataset Index File"]
	PatchSize: Annotated[Tuple[int, int] | int, "Patch Size"]
	PaddingSize: Annotated[int, "Padding Size"]
	Shuffle: Annotated[bool, "Shuffle"]  
	RandomLimit: Annotated[int, "RandomLimit"] = None
	RandomPatch: Annotated[bool, "Random Patch"] = True
	BatchDataChunkNumber: Annotated[int, "Batch Dataset Chunk Number"] = 1
	BatchSize: Annotated[int, "Batch Size"] = 1
	DropLastBatch: Annotated[bool, "Drop Last Batch"] = True
	StrideSize: Annotated[int, "Sliding Window Stride Size"] = 112
	ChannelOrder: Annotated[List[int], "Channel Order"] = None
	DataFilter: Annotated[List[str], "Data Filter By File Name Regex"] = None
	DataLoadLimit: Annotated[int, "Data Limiter"] = None
	Verbose: Annotated[bool, "Verbose"] = False

	@property
	def BatchRepeatDataSegment(self):
		return DataChunkRepeatCounts(self.BatchSize, self.BatchDataChunkNumber)
	
	@property
	def IteratorMargin(self):
		return max(self.BatchRepeatDataSegment)
	
	@property
	def FixedPatchSize(self):
		if isinstance(self.PatchSize, Tuple):
			patchX, patchY = self.PatchSize[0], self.PatchSize[1]
		else:
			patchX, patchY = self.PatchSize, self.PatchSize

		return (patchX, patchY)

	@property
	def DefaultLabelSourceConfig(self):
		return ClassConfig(names=self.ClassNames, colors=self.ClassColors, null_class=self.NullClass)



class GeoSegmentationDataset(Dataset):  # , metaclass=ABCMeta

	class DataReadType(Enum):
		IndexMetaFile = 0


	def __init__(self, config: SegmentationDatasetConfig, shared_artifacts: SharedArtifacts):
		"""Segmentasyon datasetleri için bir Base classtır."""
		self.Config = config
		self.SharedArtifacts = shared_artifacts
		self.ExpiredScenes = shared_artifacts.ExpiredScenes
		self.SourceState = shared_artifacts.AvailableInSource

		self.DatasetIndexMeta: List[DataSourceMeta]
		self.GeoDatasetCache = LimitedCache(max_size_mb=698, max_items=100)
		
		# Worker Info
		self.StartIndex = 0
		self.EndIndex = -1
		self.WorkerId = None

		# Auto Load Metadata
		self.ReadMetaData()


	def __len__(self):
		return len(self.DatasetIndexMeta)


	def __getitem__(self, idx):
		# [0, 0, 0, 0, 1, 1 ,1, 1] 
		# new_indices = list(set(self.Indices)-self.ExpiredScenes.ToSet())

		# READ SOURCE META
		_meta = self.GetIndexMetaById(idx)
		
		# READ SCENE AND CACHE
		geoDataset = self.ReadSceneDataByIndexMeta(_meta)         
		
		#! GET NEXT DATA
		data, label = self.GetNext(geoDataset, idx)

		# CHECK EXPIRING STATUS
		self.CheckGeoIteratorSceneExpiring(geoDataset)

		# UPDATE TRACKABLE STATE
		self.UpdateGeoIteratorState(_meta, geoDataset)

		return data, label


	def ReadMetaData(self, method=DataReadType.IndexMetaFile) -> DataSourceMeta:
		if method == GeoSegmentationDataset.DataReadType.IndexMetaFile:
			self.DatasetIndexMeta: List[DataSourceMeta] = GeoDataReader.ReadDatasetMetaFromIndexFile(self.Config.DatasetRootPath)
		else:
			raise ValueError(f"Bilinmeyen Yöntem: {method}. Lütfen geçerli bir veri okuma yöntemi tipi seçiniz: {self.DataReadType}")
		
		if self.Config.DataLoadLimit:
			self.DatasetIndexMeta = np.random.choice(self.DatasetIndexMeta, self.Config.DataLoadLimit, replace=False)

		return self.DatasetIndexMeta


	def LoadRasterSceneWithRasterMask(self, _data: DataSourceMeta) -> Scene:
		# Read Raster
		bands = GeoDataReader.ReadRasters(_data.DataPaths, channel_order=self.Config.ChannelOrder)
		masks = GeoDataReader.ReadRasters(_data.LabelPaths) # TODO Label tipine göre (geoJson, shapefile) okuma yapılacak (şuan labellar birer TIF dosyası ve maske şeklinde)

		# Create MultiRasterSource
		rasterSource = MultiRasterSource(bands, primary_source_idx=GeoDataReader.FindPrimarySource(bands))
		maskSource = MultiRasterSource(masks, primary_source_idx=GeoDataReader.FindPrimarySource(masks))
		
		# Create Label Source
		ssLabelSource = SemanticSegmentationLabelSource(maskSource, class_config=self.Config.DefaultLabelSourceConfig, bbox=rasterSource.bbox)
		
		# Create Scene
		scene = Scene(
			id=f"train_scene_{_data.Scene}",
			raster_source=rasterSource,
			label_source=ssLabelSource
		)

		return scene


	def CreateSlidingWindowGeoDatasetFromScene(self, scene: Scene):
		patchX, patchY = self.Config.FixedPatchSize
		return SemanticSegmentationSlidingWindowGeoDataset(
			scene=scene,
			size=(patchX, patchY),
			stride=self.Config.StrideSize, # 112
			padding=0                                     # TODO Parameter
		)


	def CreateRandomWindowGeoDatasetFromScene(self, scene: Scene):
		patchX, patchY = self.Config.FixedPatchSize
		return SemanticSegmentationRandomWindowGeoDataset(
			scene=scene,
			size_lims=(patchX, patchY+1),
			max_windows=self.Config.MaxWindowsPerScene,
			out_size=(patchX, patchY),
			padding=self.Config.PaddingSize
		)


	def ReadSceneDataByIndexMeta(self, _meta: DataSourceMeta) -> TrackableGeoIterator:
		geoDataset = self.GeoDatasetCache.Get(_meta.Scene)
		if self.Config.Verbose:
			print(f"GeoSegmentationDataset:-> index: {_meta.Index}, scene: {_meta.Scene}, pid: {os.getpid()}")
		if geoDataset is None:
			# Read Scene
			scene = self.LoadRasterSceneWithRasterMask(_meta)
			
			# Convert to GeoDataset
			if self.Config.RandomPatch:
				geoDataset = self.CreateRandomWindowGeoDatasetFromScene(scene)
				if self.Config.Verbose:
					print("Scene Length: ", len(geoDataset))
			else:
				geoDataset = self.CreateSlidingWindowGeoDatasetFromScene(scene) # Random vs Sliding

			# Wrap GeoDataset
			geoDataset = TrackableGeoIterator(geoDataset, _meta.Index, cycle=True, margin=self.Config.IteratorMargin)

			# State Tut
			self.SyncGeoIteratorState(_meta, geoDataset)

			# Cache
			self.RegisterToCache(_meta, geoDataset)
			
		return geoDataset


	def GetIndexMetaById(self, idx):
		idx %= len(self.DatasetIndexMeta)
		_meta: DataSourceMeta = self.DatasetIndexMeta[idx]
		_meta.Index = idx
		return _meta


	def CheckGeoIteratorSceneExpiring(self, geoDataset:TrackableGeoIterator):
		if geoDataset.IsExpired():
			self.ExpiredScenes.Add(geoDataset.Id)
			geoDataset.Reset()


	def RegisterToCache(self, _meta: DataSourceMeta, geoDataset:TrackableGeoIterator):
		self.GeoDatasetCache.Add(_meta.Scene, geoDataset)


	def SyncGeoIteratorState(self, _meta: DataSourceMeta, geoDataset:TrackableGeoIterator):
		if _meta.Scene in self.SourceState.keys():
			values = self.SourceState[_meta.Scene]                
			geoDataset.SetState(TrackableIteratorState(*values))    # Load Iterator State

		return _meta.Scene


	def UpdateGeoIteratorState(self, _meta: DataSourceMeta, geoDataset:TrackableGeoIterator):
		self.SourceState[_meta.Scene] = tuple(geoDataset.GetState()) # Update Storage
		
		return _meta.Scene


	def SetWorkerInfo(self, worker_id, num_workers):
		self.WorkerId = worker_id
		self.NumWorkers = num_workers
		self.segment_size = len(self.DatasetIndexMeta) // self.NumWorkers
		self.StartIndex = self.WorkerId * self.segment_size
		self.EndIndex = (self.WorkerId + 1) * self.segment_size if self.WorkerId != self.NumWorkers - 1 else len(self.__len__())


	def GetWorkerInfo(self, verbose=False):
		worker_info = torch.utils.data.get_worker_info()
		if worker_info and verbose:
			print(f"GeoSegmentationDataset:-> Worker Id: {worker_info.id+1}/{worker_info.num_workers} workers")
		return worker_info


	def GetNext(self, iterable, idx):
		data = label = None

		if self.Config.RandomPatch:
			data, label = next(iter(iterable))     # TODO: Handle => "StopIteration" Exception
		else:
			try:
				data, label = iterable[idx]
			except IndexError as e:
				print(e)

		return data, label



class BatchSamplerMode(Enum):
	Train = 0
	Val = 1
	Test = 2



class GeoSegmentationDatasetBatchSampler(Sampler):
	def __init__(self, data_source: GeoSegmentationDataset, data_split:List[float]=None, mode=BatchSamplerMode.Train):
		self.Config: SegmentationDatasetConfig = data_source.Config
		self.DataSource = data_source
		self.Indices = []
		self.Index = -1
		self.RandomLimitCounter = 0
		self.SplitRatios = data_split
		self.Mode = mode
		self.TrainIndexes = []
		self.ValIndexes = []
		self.TestIndexes = []
		self.SetIndexes()
		self.SetSplitIndexes()


	def __len__(self):
		return self.Config.RandomLimit if self.Config.RandomPatch else len(self.GetIndexes())


	def __iter__(self):
		return self


	def __next__(self):
		self.Index+=1
		self.CheckStoppingConditions()
		indexes = self.PrepareBatchIndexes()
		return indexes


	def SetIndexes(self):
		self.Indices = list(range(len(self.DataSource)))
		if self.Config.Shuffle:
			random.shuffle(self.Indices)

		return self.Indices


	def SetSplitIndexes(self):
		if not self.SplitRatios:
			self.TrainIndexes = self.Indices
			return

		# Split Indexes
		if not np.isclose(sum(self.SplitRatios), 1.0):
			raise ValueError("SplitRatios değerler toplamı 1.0 olmalıdır.")
		
		segmentLen = [round(len(self.Indices) * ratio) for ratio in self.SplitRatios]

		self.TrainIndexes = self.Indices[:segmentLen[0]]
		self.ValIndexes = self.Indices[segmentLen[0] : segmentLen[0] + segmentLen[1]]
		self.TestIndexes = self.Indices[segmentLen[0] + segmentLen[1]:]

		print(f"\nTrain: {self.TrainIndexes}\nVal: {self.ValIndexes}\nTest: {self.TestIndexes}")

	def GetIndexes(self):
		if self.Mode == BatchSamplerMode.Train:
			indices = self.TrainIndexes
		elif self.Mode == BatchSamplerMode.Val:
			indices = self.ValIndexes
		elif self.Mode == BatchSamplerMode.Test:
			indices = self.TestIndexes

		return indices


	def SetMode(self, mode: BatchSamplerMode):
		self.Mode = mode


	def CheckStoppingConditions(self):
		# TODO Multiprocess yaparken veri uzunluğu sabit kaldığından self.Index degeri artmazsa hata verebilir. Ancak batchsampler tek processte çalışır sorun olmayabilir.
		if self.Config.RandomPatch and self.Index >= self.__len__():      # Sadece Random Patch ise belirli bir epoch sayısı kadar batchler için index üretir.
			print("Random Patch Done")
			raise StopIteration
		
		
		elif len(set(self.GetIndexes()) - self.DataSource.ExpiredScenes.ToSet()) == 0:    # TODO Datasource'lar multiprocessing için bölünürse?
			if self.Config.Verbose:
				print(f"All Scenes Done, Mode: {self.Mode} ")
			
			self.DataSource.ExpiredScenes.Remove(set(self.GetIndexes()))
			self.RandomLimitCounter += 1

			raise StopIteration


	def PrepareBatchIndexes(self):
		"""
			TODO: Kaynak kalmayan indislerin yeninden düzenlenmesi gerekiyor:
			0 => 11     % 3
			1 => 12     % 4
		
			0 => State => Available: min(4, 3) = 3  
			1 => State => Available: min(4, 4) = 4
			7 => 1

			[0 0 0 0 1 1 1 1]
			[0 0 0 0 1 1 1 1]
			[0 0 0 0 1 1 1 1]
		"""	

		new_indices = list(set(self.GetIndexes())-self.DataSource.ExpiredScenes.ToSet())
	
		choices = np.random.choice(
			new_indices,
			size=len(self.Config.BatchRepeatDataSegment), # Hangi değişken
			replace=len(new_indices) < len(self.Config.BatchRepeatDataSegment)
		)

		# recovery = []
		# for id_choice, seg_count in zip(choices, self.Config.BatchRepeatDataSegment):
		# 	self.DataSource.AvailableInSource[id_choice]
		if self.Config.Verbose:
			print(f"Sampler: {choices} x {self.Config.BatchRepeatDataSegment}")
		
		return np.repeat(choices, self.Config.BatchRepeatDataSegment)



class TrackableGeoIterator():
	"""GeoSlider veya Random GeoIterator için indis takibi yapan bir sınıftır."""
	def __init__(self, iterator, id, cycle=False, margin=None):
		self.Id = id
		self.Index = -1
		self._Expired = False
		self._Cycle = cycle
		self.Iterator = iterator
		self.margin = margin


	@property
	def Margin(self):
		if 0 != self.__len__() % self.margin:
			return self.margin - (self.__len__() % self.margin)
		else:
			return 0
	

	@property
	def Available(self) -> int:
		if self.Index == -1:
			return self.__len__()
		
		return self.__len__() - ((1 + (max(self.Index, 0) % self.__len__())))


	def __iter__(self):
		return self
	

	def __next__(self):
		"""For Random GeoDataset"""
		return next(iter(self.Iterator))


	def __len__(self):
		return len(self.Iterator)


	def __getitem__(self, id):
		"""For Sliding GeoDataset"""
		self.Index += 1
		self.CheckIndex()
		# print(f"Trackable Iterator:-> Patch Index: {self.Index}-%-{self.Index % len(self.Iterator)}")
		return self.Iterator[self.Index % self.__len__()]


	def CheckIndex(self):
		if (self._Expired and not self._Cycle):
			raise IndexError
		
		self._Expired = self.Index >= self.__len__() + self.Margin - 1    # TODO Eğer alınamayacak kadar window varsa drop_last uygula veya 1 kerelik cycle yap


	def GetIndex(self):
		return self.Index


	def IsExpired(self):
		return self._Expired


	def Reset(self):
		self.Index = -1
		self._Expired = False


	def GetState(self):
		state = TrackableIteratorState(Id=self.Id, Index=self.Index, Available=self.Available, Expired=self._Expired)
		return state


	def SetState(self, state: TrackableIteratorState):
		self.Index, self._Expired = state.Index, state.Expired



def SubsetFactory(base_class: GeoSegmentationDataset, split_rates:List[float]=[0.95, 0.05]):
	DataSubsets = random_split(base_class, split_rates)
	SubsetList = []
	for subset in DataSubsets:
		class CustomSubset(base_class):
			"""Dataset objesinin uzunluğunu kullanarak istenilen oranlarda indisleri parçalara ayıran; haliyle verisetini alt gruplara bölen bir sınıftır."""
			def __init__(self):
				super().__init__(base_class.Config, base_class.SharedArtifacts)
				self.Dataset = dataset
				self.Indexes = subset.indices
				self.SplitRates = split_rates

		
			def __getitem__(self, idx):
				return self.Dataset[self.Indexes[idx]]
			
		
			def __len__(self):
				return len(self.Indexes)
		
		SubsetList += [CustomSubset()]

	return SubsetList



#%%
if "__main__" == __name__:
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


	#%%
	#! DATASET
	dataset = GeoSegmentationDataset(Config, SHARED_ARTIFACTS)


	#! DATALAODER
	customBatchSampler = GeoSegmentationDatasetBatchSampler(dataset, data_split=[0.8, 0.1, 0.1], mode=BatchSamplerMode.Train)
	DATA_LOADER = DataLoader(
		dataset,
		batch_sampler=customBatchSampler,
		num_workers=0,
		persistent_workers=False,
		pin_memory=True,
		collate_fn=CollateFN,
		# multiprocessing_context = torch.multiprocessing.get_context("spawn")
	)

	for x in range(2):
		#! SHOW RESULTS
		customBatchSampler.SetMode(BatchSamplerMode.Train)
		print("Main Process Id:", os.getpid())
		for i, (inputs, targets) in enumerate(DATA_LOADER):
			# inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
			print(f"Step: {i}", inputs.shape, targets.shape)

		print("\n")	
		customBatchSampler.SetMode(BatchSamplerMode.Test)
		print("Main Process Id:", os.getpid())
		for i, (inputs, targets) in enumerate(DATA_LOADER):
			# inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
			print(f"Step: {i}", inputs.shape, targets.shape)

		print("\n-----------------------------------------------\n")

	#! VisualizeData
	# VisualizeData(TRAIN_LOADER)
	
