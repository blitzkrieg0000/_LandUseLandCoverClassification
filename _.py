import sys
from collections import deque

class LimitedCache:
    def __init__(self, max_size_mb: int):
        self.cache = {}
        self.order = deque()
        self.max_size = max_size_mb * 1024 * 1024
        self.current_size = 0  # Mevcut bellek kullanımını takip etme

    def _get_size(self, item) -> int:
        """ Verinin bellek boyutunu hesaplar. """
        return sys.getsizeof(item)
    
    def add(self, key, value):
        item_size = self._get_size(key) + self._get_size(value)

        # Yeni elemanı eklemeden önce mevcut belleği kontrol et
        while self.current_size + item_size > self.max_size:
            if len(self.order) == 0:
                # Eğer deque boşsa, çık
                break

            # En eski anahtar-değer çiftini sil
            oldest_key = self.order.popleft()
            oldest_value = self.cache.pop(oldest_key)
            self.current_size -= (self._get_size(oldest_key) + self._get_size(oldest_value))

        # Yeni anahtar-değer çiftini ekle
        self.cache[key] = value
        self.order.append(key)
        self.current_size += item_size

    def get(self, key):
        return self.cache.get(key)

    def get_items(self):
        return {key: self.cache[key] for key in self.order}



# Örnek kullanım
cache = LimitedCache(max_size_mb=1)  # 1 MB limit

# Örnek veri ekleme
cache.add("key1", "val1")  # 1 MB'lık bir string

print(cache.get("key1"))