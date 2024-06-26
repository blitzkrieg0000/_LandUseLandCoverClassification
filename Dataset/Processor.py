import numpy as np
import torch

from Dataset.Base import BaseDatasetProcessor
from Dataset.FileReader import GeoTIFFReader
from Model.Base import ModelMeta

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class SentinelCompositeDatasetProcessor(BaseDatasetProcessor):
	def __init__(self, image_paths, mask_paths, patch_size=256, model_meta:ModelMeta=None):
		super().__init__()
		"""
			Composite Band ve Maske datasetini yönetir.
		"""
		self.image_paths = image_paths
		self.mask_paths = mask_paths
		self.patch_size = patch_size
		self.ModelMeta = model_meta
		self.data_readers:list[GeoTIFFReader] = []
		self.mask_readers:list[GeoTIFFReader] = []
		self.__OverrideModelConfig()
		self.__CreateReaders()


	def __OverrideModelConfig(self):
		"""
			Yapay zeka modeline göre ayarlar yeniden yapılandırılır.
		"""
		if self.ModelMeta is not None:
			self.patch_size = self.ModelMeta.PatchSize


	def __CreateReaders(self):
		"""
			Maske ve Band verilerini okur.
		"""
		for pth in self.image_paths:
			self.data_readers+=[GeoTIFFReader(pth, cache=True)]

		for pth in self.mask_paths:
			self.mask_readers+=[GeoTIFFReader(pth, cache=True)]


	def GetReader(self, reader, idx:int) -> GeoTIFFReader:
		return reader[idx]


	def __len__(self):
		return min(len(self.mask_paths), 16)


	def __getitem__(self, idx:int):
		buffer, window = self.GetReader(self.data_readers, idx%len(self.data_readers)).ReadRandomPatch(self.patch_size)
		mask_idx = 0 if len(self.mask_readers)==1 else idx%len(self.mask_readers)
		mask, window = self.GetReader(self.mask_readers, mask_idx).ReadRandomPatch(self.patch_size, window=window)
		return buffer, mask
