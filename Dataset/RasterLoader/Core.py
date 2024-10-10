from __future__ import annotations

import os
import random
import time
from enum import Enum
from typing import List

import numpy as np
import torch
from rastervision.core.data import (MultiRasterSource, Scene,
                                    SemanticSegmentationLabelSource)
from rastervision.pytorch_learner import (
    SemanticSegmentationRandomWindowGeoDataset,
    SemanticSegmentationSlidingWindowGeoDataset)
from torch.utils.data import Dataset, Sampler

from Dataset.RasterLoader.Default import (DataReadType,
                                          SegmentationDatasetConfig)
from Dataset.RasterLoader.FileReader import GeoDataReader
from Dataset.RasterLoader.Proxy import SharedArtifacts
from Dataset.RasterLoader.TrackableIterator import (TrackableGeoIterator,
                                                    TrackableIteratorState)
from Dataset.RasterLoader.Util import CollateFN
from Tool.Core import (ConsoleLog, GetColorsFromPalette, LimitedCache,
                       LogColorDefaults)
from Tool.Util import DataSourceMeta


class GeoSegmentationDataset(Dataset):  # , metaclass=ABCMeta
	def __init__(self, config: SegmentationDatasetConfig, shared_artifacts: SharedArtifacts):
		"""Segmentasyon datasetleri için bir Base classtır."""
		self.Config = config
		self.SharedArtifacts = shared_artifacts
		self.ExpiredScenes = shared_artifacts.ExpiredScenes
		self.SourceState = shared_artifacts.AvailableInSource

		self.DatasetIndexMeta: List[DataSourceMeta]
		self.GeoDatasetCache = LimitedCache(max_size_mb=698, max_items=100)
		
		# Auto Load Metadata
		self.ReadMetaData()

		# Worker Info
		self.StartIndex = 0
		self.EndIndex = -1
		self.SegmentSize = None
		self.WorkerId = None
		self.NumWorkers = None
		self.SetWorkerInfo()


	def __len__(self):
		return len(self.DatasetIndexMeta)


	def __getitem__(self, idx):
		ConsoleLog(f"Dataset Process Id: {os.getpid()}", LogColorDefaults.Remarkable)
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
		if method == DataReadType.IndexMetaFile:
			self.DatasetIndexMeta: List[DataSourceMeta] = GeoDataReader.ReadDatasetMetaFromIndexFile(self.Config.DatasetRootPath)
		else:
			raise ValueError(f"Bilinmeyen Yöntem: {method}. Lütfen geçerli bir veri okuma yöntemi tipi seçiniz: {DataReadType}")
		
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
			state = TrackableIteratorState(*values)                
			geoDataset.SetState(state)    # Load Iterator State

		return _meta.Scene


	def UpdateGeoIteratorState(self, _meta: DataSourceMeta, geoDataset:TrackableGeoIterator):
		self.SourceState[_meta.Scene] = tuple(geoDataset.GetState()) # Update Storage
		
		return _meta.Scene


	def GetWorkerInfo(self, verbose=False):
		worker_info = torch.utils.data.get_worker_info()
		if worker_info and verbose:
			print(f"GeoSegmentationDataset:-> Worker Id: {worker_info.id+1}/{worker_info.num_workers} workers")
		return worker_info


	def SetWorkerInfo(self):
		workerInfo = self.GetWorkerInfo()
		self.WorkerId = workerInfo.id if workerInfo else 0
		self.NumWorkers = workerInfo.num_workers if workerInfo else 1
		self.SegmentSize = len(self.DatasetIndexMeta) // max(self.NumWorkers, 1)
		self.StartIndex = self.WorkerId * self.SegmentSize
		self.EndIndex = (self.WorkerId + 1) * self.SegmentSize if self.WorkerId != self.NumWorkers - 1 else self.__len__()


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
		self.UsedIndexes = set()
		self.SetIndexes()
		self.SetSplitIndexes()


	def __len__(self):
		return self.Config.RandomLimit if self.Config.RandomPatch else len(self.GetIndexes())


	def __iter__(self):
		return self


	def __next__(self):
		self.Index+=1
		# TODO Kullanılan dataset processleri aktif durumdaysa, burayı sync et
		ConsoleLog(f"Sampler Process Id: {os.getpid()}", LogColorDefaults.Warning)
		self.CheckStoppingConditions()
		indexes = self.PrepareBatchIndexes()
		if indexes is None:
			time.sleep(0.3)           # Expire ve Used Indexleri karşılaştır.
			return self.__next__()
		
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

		# Eğer indislerin hepsi kullanılmışsa, kullanılmış indis listesini sıfırla.
		if len(set(indices) - self.UsedIndexes)==0:
			self.UsedIndexes -= set(indices)
			
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
		
		elif (set(self.GetIndexes()) - self.DataSource.ExpiredScenes.ToSet()) - self.UsedIndexes:
			...


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
		
		# Kullanılmış indisleri çıkar, böylece bir sonraki aşamaya farklı indisler dahil olur.
		# Ancak Expire olmayanları kullanma
		expiredScenes = self.DataSource.ExpiredScenes.ToSet()
		newIndices = (set(self.GetIndexes()) - expiredScenes) - self.UsedIndexes
		newIndices = list(newIndices)
		if len(newIndices)==0:
			return
		
		# Indexlerden rastgele seçim yap
		choices = np.random.choice(
			newIndices,
			size=len(self.Config.BatchRepeatDataSegment), # Hangi değişken
			replace=len(newIndices) < len(self.Config.BatchRepeatDataSegment)
		)

		# Kullanılan indisleri tüm indisler kullanılıncaya kadar kaydet
		self.UsedIndexes.update(choices.tolist())

		indexes = np.repeat(choices, self.Config.BatchRepeatDataSegment).tolist()
		
		if self.Config.Verbose:
			print(f"Sampler: {choices} x {self.Config.BatchRepeatDataSegment}")
		

		return indexes



