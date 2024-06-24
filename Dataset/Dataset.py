import torch
from torch.utils.data import Dataset

from Dataset.FileReader import ReadGeoTIFF
from Model.Base import ModelMeta
from Tool import ChangeMaskOrder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SentinelDatasetProcessor(Dataset):
	def __init__(self, image_paths, mask_paths, patch_size=256, model_meta:ModelMeta=None):
		self.image_paths = image_paths
		self.mask_paths = mask_paths
		self.patch_size = patch_size
		self.ModelMeta = model_meta
		self.data_readers:list[ReadGeoTIFF] = []
		self.mask_readers:list[ReadGeoTIFF] = []
		self.__classes = torch.tensor([1, 2, 4, 5, 7, 8, 9, 10, 11])
		self.__number_of_classes = len(self.__classes)
		self.__OverrideModelConfig()
		self.__CreateReaders()


	def __OverrideModelConfig(self):
		if self.ModelMeta is not None:
			self.patch_size = self.ModelMeta.PatchSize


	def __CreateReaders(self):
		for pth in self.image_paths:
			self.data_readers+=[ReadGeoTIFF(pth, cache=True)]

		for pth in self.mask_paths:
			self.mask_readers+=[ReadGeoTIFF(pth, cache=True)]


	def __len__(self):
		return len(self.mask_paths)*16


	def __getitem__(self, idx:int):
		buffer, window = self.data_readers[idx%len(self.data_readers)].ReadRandomPatch(self.patch_size)
		mask, window = self.mask_readers[0 if len(self.mask_readers)==1 else idx%len(self.data_readers)].ReadRandomPatch(self.patch_size, window=window)
		return buffer, mask
