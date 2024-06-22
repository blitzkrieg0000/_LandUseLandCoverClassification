import torch
from torch.utils.data import Dataset

from Dataset.FileReader import ReadGeoTIFF
from Tool import ChangeMaskOrder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SentinelPatchDataset(Dataset):
	def __init__(self, image_paths, mask_paths, patch_size=256):
		self.image_paths = image_paths
		self.mask_paths = mask_paths
		self.patch_size = patch_size
		self.data_readers:list[ReadGeoTIFF] = []
		self.mask_readers:list[ReadGeoTIFF] = []
		self.__classes = torch.tensor([1, 2, 4, 5, 7, 8, 9, 10, 11])
		self.__number_of_classes = len(self.__classes)
		self.__CreateReaders()


	def __CreateReaders(self):
		for pth in self.image_paths:
			self.data_readers+=[ReadGeoTIFF(pth, cache=True)]

		for pth in self.mask_paths:
			self.mask_readers+=[ReadGeoTIFF(pth, cache=True)]

	def Target2OneHot(self, targets):
		targets = ChangeMaskOrder(targets, self.__classes)

		#! Mask To One Hot
		targets = targets.long() # Maskeyi long yap

		# One-hot kodlamalı tensor oluştur
		one_hot_mask = torch.zeros((targets.size(0), self.__number_of_classes, targets.size(2), targets.size(3)), device=DEVICE)

		# Sınıf indekslerini one-hot kodlamalı tensor haline getir
		return one_hot_mask.scatter_(1, targets, 1)

	def __len__(self):
		return len(self.mask_paths)*16

	def __getitem__(self, idx:int):
		buffer, window = self.data_readers[idx%len(self.data_readers)].ReadRandomPatch(self.patch_size)
		mask, window = self.mask_readers[0 if len(self.mask_readers)==1 else idx%len(self.data_readers)].ReadRandomPatch(self.patch_size, window=window)
		return buffer, self.Target2OneHot(mask)
