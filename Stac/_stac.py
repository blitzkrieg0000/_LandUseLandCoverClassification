from pystac_client import Client
import rasterio
import matplotlib.pyplot as plt


catalog = Client.open("https://earth-search.aws.element84.com/v1")


bbox = (35.26, 36.74, 36.96, 37.84)  # Min lon, Min lat, Max lon, Max lat


search = catalog.search(
    bbox=bbox,
    datetime="2024-01-01/2024-08-26",     # İstediğiniz tarih aralığı
    collections=["sentinel-2-l2a"],       # Sentinel-2 L2A koleksiyonu
    query={"eo:cloud_cover": {"lt": 10}}  # Maksimum %10 bulut örtüsü
)


items = list(search.items())
if not items:
    print("Belirtilen kriterlere uygun görüntü bulunamadı.")
    exit()

item = items[0]


red_band_url = item.assets["nir"].href


with rasterio.open(red_band_url) as src:
    nir_band = src.read(1)


plt.figure(figsize=(10, 10))
plt.imshow(nir_band, cmap="Reds")
plt.title(f"Çukurova Bölgesi - {item.datetime}")
plt.axis('off')
plt.show()
