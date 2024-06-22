import torch
from Const import DATASET_CONFIG, DATASET
from FileReader import ReadGeoTIFF
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset

from Dataset.Enum import DatasetType
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



class RemoteSensingDatasetManager():
    def __init__(self): 
        ...

    def GetDataloader(self, dataset_type: DatasetType, **override_dataset_config) -> DataLoader:
        dataset = DATASET[dataset_type]
        config = DATASET_CONFIG[dataset]
        config.update(override_dataset_config)
        dataset(config["DATA"], config["MASK"], patch_size=config["PATCH_SIZE"])
        dataloader = DataLoader(dataset, batch_size=config["BATCH_SIZE"], shuffle=config["SHUFFLE"])
        return dataloader



if "__main__" == __name__:
    dataset = SentinelPatchDataset(DATA_PATH, MASK_PATH, patch_size=64)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
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