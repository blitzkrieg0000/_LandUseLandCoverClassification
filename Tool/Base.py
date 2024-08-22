from collections import deque
import random
import sys
import time
from typing import List
import numpy as np
import torch
import seaborn as sb

class EarlyStopping():
    def __init__(self, patience=5, verbose=False, delta=0, checkpoint_save_path="checkpoint.pt"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.val_loss_min = np.inf
        self.checkpoint_save_path = checkpoint_save_path


    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.SaveCheckpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping patience counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.SaveCheckpoint(val_loss, model)
            self.counter = 0


    def SaveCheckpoint(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), self.checkpoint_save_path)
        self.val_loss_min = val_loss



class LimitedCache():
    def __init__(self, max_size_mb: int=1024, max_items: int = 300):
        self.Cache = {}
        self.Order = deque()
        self.MaxSize = max_size_mb * 1024 * 1024
        self.CurrentSize = 0
        self.MaxItem = max_items
        # TODO: 1 Shared Cache ayarı yap
        # TODO: 2 Persistent Flag Ayarı Yap, Yoksa cache verileri multiprocessing aşamasında silinebilir.


    def __GetSize(self, item) -> int:
        return sys.getsizeof(item)
    

    def Get(self, key):
        return self.Cache.get(key)


    def Add(self, key, value):
        itemSize = self.__GetSize(key) + self.__GetSize(value)

        # Yeni elemanı eklemeden önce mevcut öğe sayısını kontrol et
        while len(self.Order) >= self.MaxItem or self.CurrentSize + itemSize > self.MaxSize:
            if len(self.Order) == 0:
                # Eğer deque boşsa, çık
                break

            # En eski key-value çiftini sil
            oldestKey = self.Order.popleft()
            oldestValue = self.Cache.pop(oldestKey)
            self.CurrentSize -= (self.__GetSize(oldestKey) + self.__GetSize(oldestValue))

        # Yeni key-value çiftini ekle
        self.Cache[key] = value
        self.Order.append(key)
        self.CurrentSize += itemSize



# =================================================================================================================== #
#! Functions
# =================================================================================================================== #
def ChangeMaskOrder(mask: torch.Tensor, classes: torch.Tensor):
    extra_classes = mask.unique()
    others = extra_classes[~torch.isin(extra_classes, classes)]
    other_maps = {x:0 for x in others}
    mapping = {x:i for i, x in enumerate(classes)}
    mapping.update(other_maps)

    new_mask = mask.clone()
    for old, new in mapping.items():
        new_mask[mask == old] = new
    return new_mask


def CountModelParameters(model):
    """ Count Model Parameters """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def GetTimeStampNow() -> str:
    return time.strftime("%d.%m.%Y_%H.%M.%S", time.localtime())


def GenerateRandomColors(num_colors):
    colors = []
    for _ in range(num_colors):
        color = [random.randint(0, 255) for _ in range(3)]
        colors.append(color)
    return colors


def GetColorsFromPalette(num_colors=1, pallete="husl", normalize=False):
    colors = sb.color_palette(pallete, num_colors)
    colors = np.array(colors)
    if not normalize:
        colors *= 255
        colors = colors.astype(np.uint8)

    return colors.tolist()