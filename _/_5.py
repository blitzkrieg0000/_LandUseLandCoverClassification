from multiprocessing import Manager
from typing import Any
from multipledispatch import dispatch

class SharedSetStorage():
    def __init__(self):
        self.manager = Manager()
        self.items_set = self.manager.dict()


    @dispatch(str)
    def Add(self, key: str):
        self.Add(key, key)


    @dispatch(int)
    def Add(self, key: int):
        self.Add(key, key)


    @dispatch(str, str)
    def Add(self, key: str, value: Any):
        """Öge, Storage'da Tanımlı değilse ekle. (Sette yoksa ekle)."""
        if key not in self.items_set:
            self.items_set[key] = value


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



if "__main__" == __name__:
    shared = SharedSetStorage()
    shared.Add("1", "5")
    shared.Add("2")
    shared.Add("3")
    print(shared.ToSet())
    shared.Remove(("1","2"))
    print(shared.ToSet())
    