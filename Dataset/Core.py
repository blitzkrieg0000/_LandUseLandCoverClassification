import torch
from Const import DATA_PATH, DATASET_FOR_MODEL, MASK_PATH
from Dataset.Enum import DatasetType
from FileReader import ReadGeoTIFF
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class SentinelPatchDataset(Dataset):
    def __init__(self, image_paths, mask_paths, patch_size=256):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.patch_size = patch_size
        self.data_readers:list[ReadGeoTIFF] = []
        self.mask_readers:list[ReadGeoTIFF] = []
        self.__CreateReaders()


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



class RemoteSensingDatasetManager():
    def __init__(self): 
        ...

    def GetDataloader(self, dataset_type: DatasetType) -> Dataset:
        DATASET_FOR_MODEL[dataset_type](DATA_PATH, MASK_PATH, patch_size=64)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
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