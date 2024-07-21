from __future__ import annotations

import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
from functools import reduce
from typing import Annotated, List, Tuple

import numpy as np
from matplotlib import pyplot as plt
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
from Tool.Util import (CrateDatasetIndexFile, DataSourceMeta, FilePath,
                       ReadDatasetFromIndexFile)


#%%----------------------------------------------------------------------------------------------------------------
class SegmentationDatasetConfig(BaseModel):
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
    BatchDataRepeatNumber:Annotated[int, "Batch Dataset Repeat Number"]
    BatchSize:Annotated[int, "Batch Size"]
    DropLastBatch:Annotated[bool, "Drop Last Batch"]=True



class SegmentationDataset(Dataset):
    def __init__(self):
        self.RandomPatch: SegmentationDatasetConfig
        self.ExpiredScenes: List[str]



class TrackableIterator():
    def __init__(self, iterator, id, cycle=False):
        self.Id = id
        self.Index = -1
        self._Expired = False
        self._Cycle = cycle
        self.Iterator = iterator

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
        print("Trackable Iterator:\t", "-Patch Index: ", self.Index, "-Mod Index", self.Index % len(self.Iterator))
        return self.Iterator[self.Index % self.__len__()]


    def CheckIndex(self):
        if (self._Expired and not self._Cycle):
            raise IndexError
        
        margin = max(CustomBatchSampler.BatchRepeatDataSegment)
        if 0 != self.__len__() % margin:
            margin = margin - self.__len__() % margin
        else:
            margin = 0

        self._Expired = self.Index >= self.__len__() + margin - 1    # TODO Eğer alınamayacak kadar window varsa drop_last uygula veya 1 kerelik cycle yap


    def GetIndex(self):
        return self.Index



class CustomBatchSampler(Sampler):
    BatchRepeatDataSegment = None
    def __init__(self, data_source: SegmentationDataset, config: SegmentationDatasetConfig):
        self.Shuffle = config.Shuffle
        self.RandomPatch = config.RandomPatch
        self.DataSource = data_source
        self.BatchSize = config.BatchSize
        self._DropLast = config.DropLastBatch
        self.Epoch = config.Epoch
        CustomBatchSampler.BatchRepeatDataSegment = [1]*config.BatchSize
        CustomBatchSampler.BatchRepeatDataSegment = CustomBatchSampler.RepeatedDataSegmentList(config.BatchSize, config.BatchDataRepeatNumber)
        self.Index = -1
        self.Indices = list(range(len(self.DataSource)))
        print("Batch Sampler PID:", os.getpid())
        if self.Shuffle:
            random.shuffle(self.Indices)


    @staticmethod
    def RepeatedDataSegmentList(batch_size, batch_data_repeat_number):
        """[1 1 1 1 1 1 1 1], [2 2 2 2], [3 3 2], [8]"""
        chunkSize = torch.clamp(torch.tensor(batch_data_repeat_number), 1, batch_size)
        chunks = torch.chunk(torch.arange(batch_size), chunkSize)
        return list(map(len, chunks))


    def __len__(self):
        return self.Epoch if self.RandomPatch else len(self.DataSource)


    def __iter__(self):
        return self


    def __next__(self):
        # Random Patch
        new_indices = list(set(self.Indices)-set(self.DataSource.ExpiredScenes))
    
        choices = np.random.choice(
            new_indices,
            size=len(CustomBatchSampler.BatchRepeatDataSegment),
            replace=len(new_indices) < len(CustomBatchSampler.BatchRepeatDataSegment)
        )          
    
        self.Index+=1
        if self.RandomPatch and self.Index >= self.__len__():
            print("Random Patch Done")
            raise StopIteration
        
        if self.RandomPatch and len(self.DataSource.ExpiredScenes)==len(self.DataSource)-1:
            raise StopIteration

        return np.repeat(choices, CustomBatchSampler.BatchRepeatDataSegment)  


class SpectralSegmentationDataset(SegmentationDataset):
    def __init__(self, config: SegmentationDatasetConfig):
        super().__init__()
        self.Config = config
        self.ClassConfig = ClassConfig(names=config.ClassNames, colors=config.ClassColors, null_class=config.NullClass)
        self.MaxWindowsPerScene = config.MaxWindowsPerScene
        self.PatchSize = config.PatchSize
        self.PaddingSize = config.PaddingSize
        self.Shuffle = config.Shuffle
        self.RandomPatch = config.RandomPatch
        self.GeoDatasetCache = {}
        self.DatasetIndexMeta = ReadDatasetFromIndexFile(config.DatasetRootPath)
        self.DatasetIndexMeta = self.DatasetIndexMeta[:3]
        self.ExpiredScenes = []
        self.start_idx = 0
        self.end_idx = -1


    def __len__(self):
        return len(self.DatasetIndexMeta)
    

    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        
        print(f"SpectralSegmentationDataset:-> Worker Id: {worker_info.id}/{worker_info.num_workers} workers")

        idx %= len(self.DatasetIndexMeta)
        _data: DataSourceMeta = self.DatasetIndexMeta[idx % len(self.DatasetIndexMeta)]
        geoDataset = self.GeoDatasetCache.get(_data.Scene, None)            # TODO State'i tut.
        print(f"SpectralSegmentationDataset:-> index: {idx}, scene: {_data.Scene}, pid: {os.getpid()}")
        if geoDataset is None:
            geoDataset = self.LoadData(_data)
            geoDataset = TrackableIterator(geoDataset, idx, cycle=True)
            self.GeoDatasetCache[_data.Scene] = geoDataset                    
        
        #! Get Next
        data = label = None
        if self.RandomPatch:
            data, label = next(iter(geoDataset))     # TODO: Handle => "StopIteration" Exception
        else:
            try:
                data, label = geoDataset[idx]
            except IndexError as e:
                print(e)
            self.CheckExpiring(geoDataset)
        print("---")
        return data, label


    def set_worker_info(self, worker_id, num_workers):
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.segment_size = len(self.DatasetIndexMeta) // self.num_workers
        self.start_idx = self.worker_id * self.segment_size
        self.end_idx = (self.worker_id + 1) * self.segment_size if self.worker_id != self.num_workers - 1 else len(self.data)


    def CheckExpiring(self, geoDataset:TrackableIterator):
        if geoDataset._Expired:
            self.ExpiredScenes+=[geoDataset.Id]

        if len(self.ExpiredScenes)==len(self.DatasetIndexMeta):
            self.ExpiredScenes.clear()
            raise StopIteration
        # TODO Indexable Verilerin indexlerini de sıfırla


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

        # Patch
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
                    stride = 112,
                    padding = 0                                     #TODO Parameter
                )



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
    print("\nDataloader Size:", len(DATALOADER))
    
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


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.set_worker_info(worker_id, worker_info.num_workers)


def custom_collate_fn(batch):
    data, label = zip(*batch)
    return torch.stack([d for d in data if d is not None]), torch.stack([l for l in label if l is not None])


DATASET_PATH = GetIndexDatasetPath("MiningArea01")   #"C:\\Users\\DGH\\source\\repos\\Local\\data\\GIS\\Sentinel2\\MiningArea01" # 
DATA_PATH = DATASET_PATH + f"/ab_mines/data/"
MASK_PATH = DATASET_PATH + f"/ab_mines/masks/"

dsConfig = SegmentationDatasetConfig(
    ClassNames=["background", "excavation_area"],
    ClassColors=["lightgray", "darkred"],
    NullClass="background",
    MaxWindowsPerScene=None,                        # TODO Rasterlar arasında random ve her bir raster içinde randomu ayarla
    PatchSize=(224, 224),
    PaddingSize=0,
    Shuffle=True,
    Epoch=1,
    DatasetRootPath=DATASET_PATH,
    RandomPatch=False,
    BatchDataRepeatNumber=2,
    BatchSize=8,
    DropLastBatch=True
)


dataset = SpectralSegmentationDataset(dsConfig)

customBatchSampler = CustomBatchSampler(dataset, config=dsConfig)

DATALOADER = DataLoader(
    dataset,
    batch_sampler=customBatchSampler,
    num_workers=1,
    persistent_workers=False, 
    pin_memory=True,
    # collate_fn=custom_collate_fn,
    multiprocessing_context = torch.multiprocessing.get_context("spawn")
)
print(CustomBatchSampler.BatchRepeatDataSegment)



if "__main__" == __name__:
    print("Main Process Id:", os.getpid())

    for i, (buffer, mask) in enumerate(DATALOADER):
        print("\n", f"Batch: {i}", buffer.shape, mask.shape, "\n", "-"*10)
        

    #! VisualizeData
    # VisualizeData(DATALOADER)
    
    


































