import rasterio
from matplotlib import pyplot as plt



image = rasterio.open("data/dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/data/Resample/raster/1/CompositeBandsDataset02_2023-12-01.tif")
mask = rasterio.open("data/dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/mask/raster/1/mask.tif")


fig, axs = plt.subplots(1, 2)

axs[0].imshow(image.read(1), cmap="viridis")
axs[1].imshow(mask.read(1), cmap="viridis")
plt.tight_layout()
plt.show()




