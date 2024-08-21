from collections import deque
import random
import sys
import time
from typing import List
import numpy as np
import torch


class EarlyStopping():
    def __init__(self, patience=5, verbose=False, delta=0, checkpoint_save_path="checkpoint.pt"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.val_loss_min = np.Inf
        self.checkpoint_save_path = checkpoint_save_path


    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping patience counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0


    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), self.checkpoint_save_path)
        self.val_loss_min = val_loss


class LimitedCache():
    def __init__(self, max_size_mb: int=1024, max_items: int = 300):
        self.cache = {}
        self.order = deque()
        self.max_size = max_size_mb * 1024 * 1024
        self.current_size = 0
        self.max_items = max_items


    def __GetSize(self, item) -> int:
        return sys.getsizeof(item)
    

    def Get(self, key):
        return self.cache.get(key)


    def Add(self, key, value):
        item_size = self.__GetSize(key) + self.__GetSize(value)

        # Yeni elemanı eklemeden önce mevcut öğe sayısını kontrol et
        while len(self.order) >= self.max_items or self.current_size + item_size > self.max_size:
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


# =================================================================================================================== #
#! Functions
# =================================================================================================================== #
def ChangeMaskOrder(mask: torch.Tensor, classes: torch.Tensor):
    extra_classes = classes.unique()
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
