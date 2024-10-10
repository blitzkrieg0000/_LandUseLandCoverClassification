from multiprocessing import Manager
from multiprocessing.managers import DictProxy, SyncManager
from typing import Any

from Dataset.RasterLoader.TrackableIterator import TrackableIteratorState


class SharedSetStorage():
    def __init__(self, manager: SyncManager):
        self.items_set = manager.dict()


    def Add(self, key:str, value:Any=None):
        """Öge, Storage'da Tanımlı değilse ekle. (Sette yoksa ekle)."""
        if key not in self.items_set:
            self.items_set[key] = value or key


    def Get(self):
        """Storage'dan son ögeyi al. (Sette yoksa ekle)."""
        return self.items_set.values()


    def ToSet(self):
        """Storage'deki elemanları Set'e dönüştür."""
        return set(self.items_set.values())


    def __contains__(self, item):
        """Storage'de olup olmadığını kontrol eder."""
        return item in self.items_set


    def __len__(self):
        """Storage'daki eleman sayısını verir."""
        return len(self.items_set)


    def Remove(self, items):
        """Storage'daki elemanları kaldırır."""
        for item in items:
            if item in self.items_set:
                self.items_set.pop(item, None)


    def Empty(self):
        """Storage boş mu kontrol eder."""
        return len(self.items_set)==0


    def Clear(self):
        self.items_set.clear()



class SharedArtifacts():
    _instance = None
    _manager = Manager()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    """Shared Memory aracılığı ile processler arası paylaşılan ögelerin tutulmasını sağlayan bir sınıftır."""
    def __init__(self):
        self.ExpiredScenes = SharedSetStorage(SharedArtifacts._manager)
        self.AvailableInSource: DictProxy[Any, TrackableIteratorState] = SharedArtifacts._manager.dict()
		# AvailableWorkers: DictProxy[Any, Any] = Manager().dict()



