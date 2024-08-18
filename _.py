from collections import deque
import sys


class LimitedCache():
    def __init__(self, max_size_mb: int=1024, max_items: int = 1000):
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
        print("Cache Size: ", self.current_size)



