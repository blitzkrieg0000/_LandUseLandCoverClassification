from __future__ import annotations

from abc import ABCMeta
from enum import Enum

from multiprocessing import Manager
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Dataset.FileReader import GeoDataReader
import random
from typing import Annotated, List, Set, Tuple

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

from Tool.Base import LimitedCache
from Tool.Util import (DataSourceMeta)
from multiprocessing import Manager

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


def CustomCollateFN(batch):
	data, label = zip(*batch)
	return torch.stack([d for d in data if d is not None]), torch.stack([l for l in label if l is not None])


def ShowDatasetViaVisualizer(dataset):
	vis = SemanticSegmentationVisualizer(class_names=["background", "foreground"], class_colors=["black", "white"])
	x, y = vis.get_batch(dataset, 4)
	vis.plot_batch(x, y, show=True)


def ShowRaster(raster):
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
			
			axs[3, 3].imshow(mask[bn])
			axs[3, 3].set_title("Ground Truth")
			plt.pause(1)
		
		if limit is not None and i >= limit:
			break

	plt.tight_layout()
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
class SharedQueueSet():
	def __init__(self):
		self.manager = Manager()
		self.queue = self.manager.Queue()
		self.items_set = self.manager.dict()


	def Add(self, item):
		"""Öge, Queue'da Tanımlı değilse ekle. (Sette yoksa ekle)."""
		if item not in self.items_set:
			self.queue.put(item)
			self.items_set[item] = True


	def Get(self):
		"""Queue'dan son ögeyi al. (Sette yoksa ekle)."""
		item = self.queue.get()
		self.items_set.pop(item, None)
		return item


	def ToSet(self):
		"""Queue'deki elemanları Set'e dönüştür."""
		return set(self.items_set.keys())


	def __contains__(self, item):
		"""Queue'de olup olmadığını kontrol eder."""
		return item in self.items_set


	def __len__(self):
		"""Queue'daki eleman sayısını verir."""
		return len(self.items_set)


	def Empty(self):
		"""Queue boş mu kontrol eder."""
		return self.queue.empty()


	def Clear(self):
		"""Queue'i temizler."""
		while not self.queue.empty():
			self.queue.get_nowait()
		self.items_set.clear()



class SharedArtifacts():
	ExpiredScenes = SharedQueueSet()
	AvailableInSource = Manager().dict()


class SegmentationDatasetConfig(BaseModel):
	ClassNames: Annotated[List[str], "Class Names"]
	ClassColors: Annotated[List[str| tuple], "Class Colors"]
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


class SharedSet():
    def __init__(self, shared_set):
        self.SharedSet = shared_set

    def Add(self, item):
        self.SharedSet[item] = item

    def Remove(self, item):
        if item in self.SharedSet:
            del self.SharedSet[item]

    def IsIn(self, item):
        return item in self.SharedSet


class CustomSubset(Dataset):
	def __init__(self, dataset: Dataset, split_rates: List[float]=[0.0009, 0.05]) -> None:
		self.Dataset = dataset
		self.indices = []
		self.SplitRates = split_rates

	
	def CreateSubsets(self):
		self.DataSubsets = random_split(self.Dataset, self.SplitRates)
		self.indices = self.DataSubsets[0].indices


	def __getitem__(self, idx):
		if isinstance(idx, list):
			return self.Dataset[[self.indices[i] for i in idx]]
		return self.Dataset[self.indices[idx]]
	

	def __len__(self):
		return len(self.indices)



class SegmentationDatasetBase(Dataset, metaclass=ABCMeta):

	class DataReadType(Enum):
		IndexMetaFile=0


	def __init__(self, config: SegmentationDatasetConfig, shared_artifacts: SharedArtifacts):
		"""Segmentasyon datasetleri için bir Base classtır."""
		self.Config = config
		self.ExpiredScenes = shared_artifacts.ExpiredScenes
		self.AvailableInSource = shared_artifacts.AvailableInSource

		self.DatasetIndexMeta: List[DataSourceMeta]
		self.GeoDatasetCache = LimitedCache(max_size_mb=512, max_items=100)
		
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

		return data, label


	def ReadMetaData(self, method=DataReadType.IndexMetaFile) -> DataSourceMeta:
		if method == SegmentationDatasetBase.DataReadType.IndexMetaFile:
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


	def ReadSceneDataByIndexMeta(self, _data: DataSourceMeta) -> TrackableGeoIterator:
		geoDataset = self.GeoDatasetCache.Get(_data.Scene)
		print(f"SpectralSegmentationDataset:-> index: {_data.Index}, scene: {_data.Scene}, pid: {os.getpid()}")
		if geoDataset is None:
			# Read Scene
			scene = self.LoadRasterSceneWithRasterMask(_data)
			
			# Convert to GeoDataset
			if self.Config.RandomPatch:
				geoDataset = self.CreateRandomWindowGeoDatasetFromScene(scene)
				print(len(geoDataset))
			else:
				geoDataset = self.CreateSlidingWindowGeoDatasetFromScene(scene) # Random vs Sliding

			# Wrap GeoDataset
			geoDataset = TrackableGeoIterator(geoDataset, _data.Index, cycle=True, margin=self.Config.IteratorMargin)

			# Cache
			self.GeoDatasetCache.Add(_data.Scene, geoDataset)       

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

		self.AvailableInSource[geoDataset.Id] = geoDataset.Available


	def SetWorkerInfo(self, worker_id, num_workers):
		self.WorkerId = worker_id
		self.NumWorkers = num_workers
		self.segment_size = len(self.DatasetIndexMeta) // self.NumWorkers
		self.StartIndex = self.WorkerId * self.segment_size
		self.EndIndex = (self.WorkerId + 1) * self.segment_size if self.WorkerId != self.NumWorkers - 1 else len(self.__len__())


	def GetWorkerInfo(self, verbose=False):
		worker_info = torch.utils.data.get_worker_info()
		if worker_info and verbose:
			print(f"SpectralSegmentationDataset:-> Worker Id: {worker_info.id+1}/{worker_info.num_workers} workers")
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



class SegmentationBatchSamplerBase(Sampler):
	def __init__(self, data_source: SegmentationDatasetBase):
		self.Config: SegmentationDatasetConfig = data_source.Config
		self.DataSource = data_source
		self.Indices = list(range(len(self.DataSource)))
		self.Index = -1
		self.RandomLimitCounter = 0

		if self.Config.Shuffle:
			random.shuffle(self.Indices)


	def __len__(self):
		return self.Config.RandomLimit if self.Config.RandomPatch else len(self.DataSource)


	def __iter__(self):
		return self


	def __next__(self):
		self.Index+=1
		self.CheckStoppingConditions()
		indexes = self.PrepareBatchIndexes()
		return indexes


	def CheckStoppingConditions(self):
		# TODO Multiprocess yaparken veri uzunluğu sabit kaldığından self.Index degeri artmazsa hata verebilir. Ancak batchsampler tek processte çalışır sorun olmayabilir.
		if self.Config.RandomPatch and self.Index >= self.__len__():      # Sadece Random Patch ise belirli bir epoch sayısı kadar batchler için index üretir.
			print("Random Patch Done")
			raise StopIteration
		
		elif len(self.DataSource.ExpiredScenes) >= len(self.DataSource):    # TODO Datasource'lar multiprocessing için bölünürse?
			self.DataSource.ExpiredScenes.Clear()
			self.RandomLimitCounter += 1

			raise StopIteration


	def PrepareBatchIndexes(self):
		# 0 => 11     % 3
		# 1 => 12     % 4
	
		# 0 => State => Available: min(4, 3) = 3  
		# 1 => State => Available: min(4, 4) = 4
		#7 => 1

		# [0 0 0 0 1 1 1 1]
		# [0 0 0 0 1 1 1 1]
		# [0 0 0 0 1 1 1 1]

		new_indices = list(set(self.Indices)-self.DataSource.ExpiredScenes.ToSet())
	

		choices = np.random.choice(
			new_indices,
			size=len(self.Config.BatchRepeatDataSegment), # Hangi değişken
			replace=len(new_indices) < len(self.Config.BatchRepeatDataSegment)
		)

		# recovery = []
		# for id_choice, seg_count in zip(choices, self.Config.BatchRepeatDataSegment):
		# 	self.DataSource.AvailableInSource[id_choice]

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



class SegmentationBatchSampler(SegmentationBatchSamplerBase):
	def __init__(self, data_source: SegmentationDatasetBase):
		super().__init__(data_source)
	

	def __iter__(self):
		return super().__iter__()


	def __next__(self):
		indexes = super().__next__()
		return indexes



class SpectralSegmentationDataset(SegmentationDatasetBase):
	def __init__(self, config: SegmentationDatasetConfig, shared_artifacts: SharedArtifacts):
		super().__init__(config, shared_artifacts)


	def __getitem__(self, idx):
		data, label = super().__getitem__(idx)
		return data, label






if "__main__" == __name__:
	DATASET_PATH = "data/dataset/SeasoNet/"
	SHARED_ARTIFACTS=SharedArtifacts()

	dsConfig = SegmentationDatasetConfig(
		ClassNames=["background", "excavation_area"],
		ClassColors=["lightgray", "darkred"],
		NullClass="background",
		MaxWindowsPerScene=None,                         # TODO Rasterlar arasında random ve her bir raster içinde randomu ayarla
		PatchSize=(120, 120),
		PaddingSize=0,
		Shuffle=True,
		DatasetRootPath=DATASET_PATH,
		RandomLimit=0,
		RandomPatch=False,
		BatchDataChunkNumber=16,
		BatchSize=16,
		DropLastBatch=True,
		# ChannelOrder=[1,2,3,7],
		DataFilter=[".*_10m_RGB", ".*_10m_IR", ".*_20m"],
		DataLoadLimit=20
	)

	dataset = SpectralSegmentationDataset(dsConfig, SHARED_ARTIFACTS)
	
	customBatchSampler = SegmentationBatchSampler(dataset)
	
	TRAIN_LOADER = DataLoader(
		dataset,
		batch_sampler=customBatchSampler,
		num_workers=1,
		persistent_workers=False,
		pin_memory=True,
		collate_fn=CustomCollateFN,
		# multiprocessing_context = torch.multiprocessing.get_context("spawn")
	)

	# print("Dataset Len:", len(TRAIN_LOADER))
	# for step, (inputs, targets) in enumerate(TRAIN_LOADER):
	# 	print(f"=>Step: {step}", inputs.shape, targets.shape)

	#! VisualizeData
	VisualizeData(TRAIN_LOADER)

	exit()

	valRatio = 0.0009
	testRatio = 0.05
	trainset, valset, testset = random_split(dataset, [1-testRatio-valRatio, valRatio, testRatio])
	print(len(trainset), len(valset), len(testset))

	customBatchSampler = SegmentationBatchSampler(dataset)

	TRAIN_LOADER = DataLoader(
		trainset,
		batch_sampler=customBatchSampler,
		num_workers=2,
		persistent_workers=False, 
		pin_memory=True,
		collate_fn=CustomCollateFN,
		# multiprocessing_context = torch.multiprocessing.get_context("spawn")
	)
	
	VAL_LOADER = DataLoader(valset, batch_size=1)


	print("Main Process Id:", os.getpid())
	# for i, (inputs, targets) in enumerate(TRAIN_LOADER):
	#     inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
	#     print("\n", "-"*10)
	#     print(f"Batch: {i}", inputs.shape, targets.shape)
	#     print("-"*10, "\n")
	#     print(f"Batch: {i}")

	#! VisualizeData
	VisualizeData(VAL_LOADER)
	
	
