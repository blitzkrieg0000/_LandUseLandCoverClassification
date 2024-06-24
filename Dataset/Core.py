import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from Dataset.Const import DATASET
from Dataset.Dataset import DatasetProcessor, SentinelDatasetProcessor
from Dataset.Enum import DatasetType

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RemoteSensingDatasetManager():
	def __init__(self, dataset_processor: DatasetProcessor): 
		self.DatasetProcessor = dataset_processor

	def GetDataloader(self, dataset_type: DatasetType, override_dataset_config={}) -> DataLoader:
		config = DATASET[dataset_type]
		config.update(override_dataset_config)
		_dataset = self.DatasetProcessor(config["DATA"], config["MASK"], patch_size=config["PATCH_SIZE"])
		dataloader = DataLoader(_dataset, batch_size=config["BATCH_SIZE"], shuffle=config["SHUFFLE"])
		return dataloader


if "__main__" == __name__:
	dataloader = RemoteSensingDatasetManager(SentinelDatasetProcessor).GetDataloader(DatasetType.Cukurova_IO_LULC)
	buffer, mask = next(iter(dataloader))

	# Show Patches
	fig, axs = plt.subplots(4, 4, figsize=(12, 12))
	for i in range(16):
		if i<buffer.shape[1]:
			axs[i%4, i//4].imshow(buffer[0, i].cpu().numpy(), cmap="gray")  # Grayscale olarak görselleştirme
		axs[i%4, i//4].axis("off")
	axs[3, 3].imshow(mask[0, 0])
	plt.tight_layout()

	# Show Patches
	fig, axs = plt.subplots(4, 4, figsize=(12, 12))
	for i in range(16):
		if i<buffer.shape[1]:
			axs[i%4, i//4].imshow(buffer[1, i].cpu().numpy(), cmap="gray")  # Grayscale olarak görselleştirme
		axs[i%4, i//4].axis("off")
	axs[3, 3].imshow(mask[1, 0])
	plt.tight_layout()
	plt.show()