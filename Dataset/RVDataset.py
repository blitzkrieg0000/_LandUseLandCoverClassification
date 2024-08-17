from __future__ import annotations

import os
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
from collections import deque
from functools import reduce
from typing import Annotated, List, Set, Tuple

import numpy as np
import torch
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    BatchDataChunkNumber:Annotated[int, "Batch Dataset Repeat Number"]
    BatchSize:Annotated[int, "Batch Size"]
    DropLastBatch:Annotated[bool, "Drop Last Batch"]=True
    StrideSize:Annotated[int, "Sliding Window Stride Size"]=112
    ChannelOrder:Annotated[List[int], "Channel Order"]=None
    DataFilter:Annotated[List[str], "Data Filter"]=None


class SegmentationDataset(Dataset):
    def __init__(self):
        self.GeoDatasetCache: LimitedCache
        self.RandomPatch: SegmentationDatasetConfig
        self.ExpiredScenes: Set[str]



class LimitedCache():
    def __init__(self, max_size_mb: int):
        self.cache = {}
        self.order = deque()
        self.max_size = max_size_mb * 1024 * 1024
        self.current_size = 0

    def __GetSize(self, item) -> int:
        return sys.getsizeof(item)
    
    def Add(self, key, value):
        item_size = self.__GetSize(key) + self.__GetSize(value)

        # Yeni elemanı eklemeden önce mevcut belleği kontrol et
        while self.current_size + item_size > self.max_size:
            if len(self.order) == 0:
                # Eğer deque boşsa, çık
                break

            # En eski key-value çiftini sil
            oldest_key = self.order.popleft()
            oldest_value = self.cache.pop(oldest_key)
            self.current_size -= (self.__GetSize(oldest_key) + self.__GetSize(oldest_value))

        # Yeni key-value çiftini ekle
        self.cache[key] = value
        self.order.append(key)
        self.current_size += item_size

    def Get(self, key):
        return self.cache.get(key)

    def GetItems(self):
        return {key: self.cache[key] for key in self.order}


class TrackableIterator():
    """GeoSlider veya Random GeoIterator için indis takibi yapan bir sınıftır."""
    def __init__(self, iterator, id, cycle=False, margin=None):
        self.Id = id
        self.Index = -1
        self._Expired = False
        self._Cycle = cycle
        self.Iterator = iterator
        self.margin = margin or max(CustomBatchSampler.BatchRepeatDataSegment)

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


    @property
    def Margin(self):
        if 0 != self.__len__() % self.margin:
            return self.margin - (self.__len__() % self.margin)
        else:
            return 0


    def CheckIndex(self):
        if (self._Expired and not self._Cycle):
            raise IndexError
        
        self._Expired = self.Index >= self.__len__() + self.Margin - 1    # TODO Eğer alınamayacak kadar window varsa drop_last uygula veya 1 kerelik cycle yap


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
        self.Index = -1
        self.EpochCounter = 0
        self.Indices = list(range(len(self.DataSource)))
        CustomBatchSampler.RepeatedDataSegmentList(config.BatchSize, config.BatchDataChunkNumber)


    @staticmethod
    def RepeatedDataSegmentList(batch_size, batch_data_chunk_number):
        """[1 1 1 1 1 1 1 1], [2 2 2 2], [4 4], [3 3 2], [8]"""
        chunkSize = torch.clamp(torch.tensor(batch_data_chunk_number), 1, batch_size)
        chunks = torch.chunk(torch.arange(batch_size), chunkSize) 
        CustomBatchSampler.BatchRepeatDataSegment = list(map(len, chunks))


    def __len__(self):
        return self.Epoch if self.RandomPatch else len(self.DataSource)


    def __iter__(self):
        if self.Shuffle:
            random.shuffle(self.Indices)
        return self


    def __next__(self):
        if len(self.DataSource.ExpiredScenes)>=len(self.DataSource):    # TODO Datasource'lar multiprocessing için bölünürse?
            self.DataSource.ExpiredScenes.clear()
            self.EpochCounter += 1

            if self.EpochCounter >= self.Epoch:
                # self.EpochCounter = 0
                raise StopIteration


        # Random Patch
        new_indices = list(set(self.Indices)-self.DataSource.ExpiredScenes)
    
        choices = np.random.choice(
            new_indices,
            size=len(CustomBatchSampler.BatchRepeatDataSegment),
            replace=len(new_indices) < len(CustomBatchSampler.BatchRepeatDataSegment)
        )          
    
        self.Index+=1

        # Sadece Random Patch ise belirli bir epoch sayısı kadar batchler için index üretir.
        if self.RandomPatch and self.Index >= self.__len__():
            print("Random Patch Done")
            raise StopIteration
        
        return np.repeat(choices, CustomBatchSampler.BatchRepeatDataSegment)  


class SpectralSegmentationDataset(SegmentationDataset):
    def __init__(self, config: SegmentationDatasetConfig):
        super().__init__()
        self.Config = config
        self.ClassConfig = ClassConfig(names=config.ClassNames, colors=config.ClassColors, null_class=config.NullClass)
        self.MaxWindowsPerScene = config.MaxWindowsPerScene
        self.StrideSize = config.StrideSize
        self.PatchSize = config.PatchSize
        self.PaddingSize = config.PaddingSize
        self.Shuffle = config.Shuffle
        self.RandomPatch = config.RandomPatch
        self.GeoDatasetCache = LimitedCache(max_size_mb=1024)
        self.DatasetIndexMeta: List[DataSourceMeta] = ReadDatasetFromIndexFile(config.DatasetRootPath)
        # self.DatasetIndexMeta = self.DatasetIndexMeta[:3]
        self.ExpiredScenes = set()
        self.start_idx = 0
        self.end_idx = -1


    def __len__(self):
        return len(self.DatasetIndexMeta)
    

    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            print(f"SpectralSegmentationDataset:-> Worker Id: {worker_info.id}/{worker_info.num_workers} workers")

        # READ SOURCE META
        idx %= len(self.DatasetIndexMeta)
        _data: DataSourceMeta = self.DatasetIndexMeta[idx]

        # READ SCENE AND CACHE
        geoDataset = self.GeoDatasetCache.Get(_data.Scene)            # TODO State'i tut.
        print(f"SpectralSegmentationDataset:-> index: {idx}, scene: {_data.Scene}, pid: {os.getpid()}")
        if geoDataset is None:
            geoDataset = self.LoadData(_data)
            geoDataset = TrackableIterator(geoDataset, idx, cycle=True)
            self.GeoDatasetCache.Add(_data.Scene, geoDataset)              
        
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
            self.ExpiredScenes.add(geoDataset.Id)
            # Reset Index
            geoDataset.Index = -1
            geoDataset._Expired = False

        # if len(self.ExpiredScenes)==self.__len__():
        #     self.ExpiredScenes.clear()
        #     raise StopIteration
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


    def SortByPatterns(self, path_list, data_filter):
        def match_priority(path):
            for i, pattern in enumerate(data_filter):
                if re.search(pattern, path):
                    return i
            return len(data_filter)
        
        return sorted(path_list, key=match_priority)


    def ReadRaster(self, file_path=List[FilePath], allow_streaming=False, raster_transformers=[], channel_order=None, bbox=None):
        paths = [fp.Path for fp in file_path]
        
        if self.Config.DataFilter:
            paths = self.SortByPatterns(paths, self.Config.DataFilter)
        
        rasters = []
        for path in paths:
            raster = RasterioSource(
                        path,
                        allow_streaming=allow_streaming,
                        raster_transformers=raster_transformers,
                        channel_order=channel_order,
                        bbox=bbox
                    )
            
            rasters+=[raster]
        return rasters


    def LoadData(self, _data: DataSourceMeta):
        # Read Raster
        bands = self.ReadRaster(_data.DataPaths, channel_order=self.Config.ChannelOrder)
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
                    stride = self.StrideSize, # 112
                    padding = 0                                     # TODO Parameter
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
                axs[i%4, i//4].imshow(buffer[bn, i].cpu().numpy(), cmap="viridis")
                axs[i%4, i//4].set_title(f"Band {i+1}")
        
        axs[2, 3].imshow(mask[bn].cpu().numpy(), cmap="viridis")
        axs[2, 3].set_title("Ground Truth")
        axs[3, 3].imshow(predicted[bn].cpu().numpy(), cmap="viridis")
        axs[3, 3].text(0, 0, "Predicted", fontsize=12, color="blue", weight="bold")

    plt.tight_layout()
    plt.show()


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.set_worker_info(worker_id, worker_info.num_workers)


def custom_collate_fn(batch):
    data, label = zip(*batch)
    return torch.stack([d for d in data if d is not None]), torch.stack([l for l in label if l is not None])


# os.environ["DATA_INDEX_FILE"] ="data/dataset/.index"
# DATASET_PATH = GetIndexDatasetPath("LULC_IO_10m")

DATASET_PATH = "data/dataset/SeasoNet/"

dsConfig = SegmentationDatasetConfig(
    ClassNames=["background", "excavation_area"],
    ClassColors=["lightgray", "darkred"],
    NullClass="background",
    MaxWindowsPerScene=None,                        # TODO Rasterlar arasında random ve her bir raster içinde randomu ayarla
    PatchSize=(120, 120),
    PaddingSize=0,
    Shuffle=True,
    Epoch=5,
    DatasetRootPath=DATASET_PATH,
    RandomPatch=False,
    BatchDataChunkNumber=8,
    BatchSize=8,
    DropLastBatch=True,
    # ChannelOrder=[1,2,3,7],
    DataFilter=[".*_10m", ".*_20m", ".*_IR"]
)


dataset = SpectralSegmentationDataset(dsConfig)

customBatchSampler = CustomBatchSampler(dataset, config=dsConfig)


DATALOADER = DataLoader(
    dataset,
    batch_sampler=customBatchSampler,
    num_workers=0,
    persistent_workers=False, 
    pin_memory=True,
    collate_fn=custom_collate_fn,
    # multiprocessing_context = torch.multiprocessing.get_context("spawn")
)

if "__main__" == __name__:
    print("Main Process Id:", os.getpid())

    # for i, (inputs, targets) in enumerate(DATALOADER):
    #     inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
    #     print("\n", "-"*10)
    #     print(f"Batch: {i}", inputs.shape, targets.shape)
    #     print("-"*10, "\n")
    #     print(f"Batch: {i}")

    #! VisualizeData
    VisualizeData(DATALOADER)
    
    


































