import random

import numpy as np
import rasterio
import rasterio.windows
import torch


class GeoTIFFReader():
    """
        GeoTIF/TIF/TIFF okuma işlemlerini gerçekleştirir.
    """
    def __init__(self, filepath, cache=True):
        self.filepath = filepath
        self.file_reader = None
        self.__cache = cache
        self.__band_count = 1
        self.__raster_width = 0
        self.__raster_height = 0
        self.__file=None
        self.__OpenFile()

    def __del__(self):
        if self.file_reader:
            self.file_reader.close()
    
    
    def __OpenFile(self):
        self.file_reader = rasterio.open(self.filepath)
        self.__band_count = self.file_reader.count
        self.__raster_height, self.__raster_width = self.file_reader.height, self.file_reader.width

        if self.__cache:
            self.__file=self.file_reader.read()
            self.__file=np.array(self.__file, dtype=np.float32)
            self.file_reader.close()
            self.file_reader = None

        return self


    def ReadRandomPatch(self, patch_size=256, window=None) -> torch.Tensor:
        if window is None:
            # Patch için pencere konumunu belirle
            x = random.randint(0, self.__raster_width - patch_size)
            y = random.randint(0, self.__raster_height - patch_size)
            window = rasterio.windows.Window(x, y, patch_size, patch_size)

        # Patch bufferı oluştur.
        buffer = np.zeros((self.__band_count, patch_size, patch_size), dtype=np.float32)
        
        for i in range(self.__band_count):
            # Read Patch
            if self.__cache:
                patch = self.__file[i, window.row_off:window.row_off+patch_size, window.col_off:window.col_off+patch_size]
            else:
                patch = self.file_reader.read(i+1, window=window)

            # Register Patch
            patch = np.array(patch, dtype=np.float32)
            
            buffer[i] = patch

        return torch.tensor(buffer, dtype=torch.float32), window

