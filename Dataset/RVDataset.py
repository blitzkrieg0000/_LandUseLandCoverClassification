import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
from functools import reduce
from typing import Annotated, List, Tuple

from matplotlib import pyplot as plt
import numpy as np
from pydantic import BaseModel
from rastervision.core.data import (ClassConfig, MultiRasterSource,
                                    RasterioSource, Scene,
                                    SemanticSegmentationLabelSource)
from rastervision.pytorch_learner import (
    SemanticSegmentationRandomWindowGeoDataset,
    SemanticSegmentationSlidingWindowGeoDataset,
    SemanticSegmentationVisualizer)
from torch.utils.data import DataLoader, Dataset, Sampler

from Tool.DataStorage import GetIndexDatasetPath
from Tool.Util import DataSourceMeta, FilePath, ReadDatasetFromIndexFile





#%%----------------------------------------------------------------------------------------------------------------
class SpectralSegmentationDatasetConfig(BaseModel):
    ClassNames: Annotated[List[str], "Class Names"]
    ClassColors: Annotated[List[str], "Class Colors"]
    NullClass: Annotated[str, "Null Class"]
    MaxWindowsPerScene: Annotated[int | float | None, "Max Windows Per Scene"]
    DatasetRootPath:Annotated[str, "Dataset Index File"]
    PatchSize:Annotated[Tuple[int, int] | int, "Patch Size"]
    PaddingSize:Annotated[int, "Padding Size"]
    Shuffle:Annotated[bool, "Shuffle"]  
    Epoch:Annotated[int, "Epoch"]=None
    RandomPatch:Annotated[bool, "Random Patch"]=True


class TrackingIterator():
    def __init__(self, iterator, limit=None, cycle=False):
        self.iterator = iterator
        self.Limit = limit or len(iterator)
        self.index = -1
        self._Expired = False
        self._Cycle = cycle
        if self.Limit > len(iterator):
            self._Cycle = True

    def CheckIndex(self):
        self.index += 1

        if (self._Expired and not self._Cycle) or (self.index >= self.Limit):
            raise IndexError

        self._Expired = self.index >= len(self.iterator)-1

    def __iter__(self):
        return self

    def __len__(self):
        return self.Limit
    
    def __getitem__(self, idx):
        self.CheckIndex()
        print("patch index: ", self.index % len(self.iterator))
        return self.iterator[self.index % len(self.iterator)]

    def __next__(self):
        return next(iter(self.iterator))

    def GetIndex(self):
        return self.index


class CustomBatchSampler(Sampler):
    def __init__(self, data_source, batch_size: int, batch_data_repeat_number: int=1, shuffle: bool=False):
        self.Shuffle = shuffle
        self.DataSource = data_source
        self.BatchSize = batch_size
        self.BatchRepeatDataSegment = [1]*batch_size
        self.RepeatedDataSegmentList(batch_size, batch_data_repeat_number)


    def RepeatedDataSegmentList(self, batch_size, batch_data_repeat_number):
        batch_data_repeat_number = np.clip(batch_data_repeat_number, 1, batch_size)   # min(max(1, batch_data_repeat_number), batch_size) 
        segments = np.array([batch_data_repeat_number]*(batch_size // batch_data_repeat_number) + [batch_size % batch_data_repeat_number])
        self.BatchRepeatDataSegment = segments[segments != 0] 


    def __len__(self):
        """Random veri işleniyorsa, Epoch sayısı kadar veri oluşsun."""
        print("Datasource Length: ", len(self.DataSource))
        return len(self.DataSource)


    def __iter__(self):
        stride = len(self.BatchRepeatDataSegment) # [1 1 1 1 1 1 1 1] | [2 2 2 2] | [3 3 2] | [8]
        indices = list(range(self.__len__()))
        if self.Shuffle:
            random.shuffle(indices)
        for start_idx in range(0, self.__len__(), stride):
            indexes = indices[start_idx:start_idx + stride]
            yield np.repeat(indexes, self.BatchRepeatDataSegment[:len(indexes)])



class SpectralSegmentationDataset(Dataset):
    def __init__(self, config: SpectralSegmentationDatasetConfig):
        self.ClassConfig = ClassConfig(names=config.ClassNames, colors=config.ClassColors, null_class=config.NullClass)
        self.MaxWindowsPerScene = config.MaxWindowsPerScene
        self.PatchSize = config.PatchSize
        self.PaddingSize = config.PaddingSize
        self.Shuffle = config.Shuffle
        self.Epoch = config.Epoch
        self.RandomPatch = config.RandomPatch
        self.GeoDatasetCache = {}
        self.DatasetIndexMeta = ReadDatasetFromIndexFile(config.DatasetRootPath)
        self.ShuffleDataset()


    def __len__(self):
        return self.Epoch or len(self.DatasetIndexMeta)


    def __getitem__(self, idx):
        idx %= len(self.DatasetIndexMeta)
        _data = self.DatasetIndexMeta[idx % len(self.DatasetIndexMeta)]
        geoDataset = self.GeoDatasetCache.get(_data.Scene, None)
        print("index: ", idx, "scene: ", _data.Scene, "pid", os.getpid())
        if geoDataset is None:
            geoDataset = self.LoadData(_data)
            geoDataset = TrackingIterator(geoDataset)
            self.GeoDatasetCache[_data.Scene] = geoDataset
        
        #! Get Next
        # data, label = geoDataset[idx]
        if self.RandomPatch:
            data, label = next(iter(geoDataset))
        else:
            data, label = geoDataset[idx]

        return data, label


    def FindPrimarySource(self, bands: List[DataSourceMeta]):
        """
            MultiRasterSource'un birden fazla bandı stack'lerken kullanacağı referans band'ın index numarasını arar.
            En büyük shape'e sahip bandın index numarasını döndürür.
        """
        reference_band_index=0
        band_size=0
        for band_index, band in enumerate(bands):
            size = reduce(lambda x, y: x * y, band.shape[:-1])
            if size >= band_size:
                band_size = size
                reference_band_index = band_index

        return reference_band_index


    def ShuffleDataset(self):
        if self.Shuffle:
            random.shuffle(self.DatasetIndexMeta) # torch.randperm


    def ReadRaster(self, file_path=List[FilePath], allow_streaming=False, raster_transformers=[], channel_order=None, bbox=None):
        return [
            RasterioSource(
            fp.Path,
            allow_streaming=allow_streaming,
            raster_transformers=raster_transformers,
            channel_order=channel_order,
            bbox=bbox
        ) for fp in file_path]


    def LoadData(self, _data: DataSourceMeta):
        # Read Raster
        bands = self.ReadRaster(_data.DataPaths)
        # TODO Label tipine göre (geoJson, shapefile) okuma yapılacak (şuan labellar birer TIF dosyası ve maske şeklinde)
        masks = self.ReadRaster(_data.LabelPaths)

        # Create MultiRasterSource
        rasterSource = MultiRasterSource(bands, primary_source_idx=self.FindPrimarySource(bands))
        maskSource = MultiRasterSource(masks, primary_source_idx=self.FindPrimarySource(masks))
        
        # Create Label Source
        ssLabelSource = SemanticSegmentationLabelSource(maskSource, class_config=self.ClassConfig, bbox=rasterSource.bbox)
        
        # Create Scene
        scene = Scene(
            id=f"train_scene_{_data.Scene}",
            raster_source=rasterSource,
            label_source=ssLabelSource
        )

        # RandomPatch
        if isinstance(self.PatchSize, Tuple):
            patchX, patchY = self.PatchSize[0], self.PatchSize[1]
        else:
            patchX, patchY = self.PatchSize, self.PatchSize
        
        if self.RandomPatch:
            #! 1-RANDOM
            return SemanticSegmentationRandomWindowGeoDataset(
                scene = scene,
                size_lims = (patchX, patchY+1),
                max_windows = self.MaxWindowsPerScene,
                out_size = (patchX, patchY),
                padding = self.PaddingSize
            )
        else:
            #! 2-SEQUENCE
            return SemanticSegmentationSlidingWindowGeoDataset(
                    scene = scene,
                    size = (patchX, patchY),
                    stride = 112
                )


def ShowDatasetViaVisualizer(dataset):
    vis = SemanticSegmentationVisualizer(class_names=["background", "foreground"], class_colors=["black", "white"])
    x, y = vis.get_batch(dataset, 4)
    vis.plot_batch(x, y, show=True)


def ShowRaster(raster):
    fig, ax = plt.subplots(4, 4, figsize=(7, 7))
    for i in range(12):
        ax[i%4, i//4].matshow(raster[:, :, [i]], cmap="plasma")
        ax[i%4, i//4].axis("off")

    plt.show()


def GetNext(TRAIN_DATALOADER):
    buffer, mask = next(iter(TRAIN_DATALOADER))
    for batch in range(buffer.shape[0]):
        # Show Patches
        fig, axs = plt.subplots(4, 4, figsize=(12, 12))
        for i in range(16):
            if i<buffer.shape[1]:
                axs[i%4, i//4].imshow(buffer[batch, i].numpy(), cmap="gray")  # Grayscale olarak görselleştirme
            axs[i%4, i//4].axis("off")
        axs[3, 3].imshow(mask[batch])
        plt.tight_layout()
        plt.show()



if "__main__" == __name__:
    DATASET_PATH = GetIndexDatasetPath("MiningArea01")
    DATA_PATH = DATASET_PATH + f"/ab_mines/data/"
    MASK_PATH = DATASET_PATH + f"/ab_mines/masks/"

    dsConfig = SpectralSegmentationDatasetConfig(
        ClassNames=["background", "excavation_area"],
        ClassColors=["lightgray", "darkred"],
        NullClass="background",
        MaxWindowsPerScene=None,                        # TODO Rasterlar arasında random ve her bir raster içinde randomu ayarla
        PatchSize=(224, 224),
        PaddingSize=0,
        Shuffle=False,
        Epoch=100,
        DatasetRootPath=DATASET_PATH,
        RandomPatch=True
    )
    
    print("parent pid", os.getpid())
    dataset = SpectralSegmentationDataset(dsConfig)
    customBatchSampler = CustomBatchSampler(dataset, batch_size=8, batch_data_repeat_number=2, shuffle=T)
    DATALOADER = DataLoader(dataset, batch_sampler=customBatchSampler, num_workers=0, persistent_workers=False, pin_memory=True)
    
    print("Dataloader Size:", len(DATALOADER))
    # GetNext(DATALOADER)
    # GetNext(DATALOADER)

    for i, (buffer, mask) in enumerate(DATALOADER):
        print(i, buffer.shape, mask.shape)