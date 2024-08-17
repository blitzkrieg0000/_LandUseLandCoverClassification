import rasterio
from matplotlib import pyplot as plt

scene_path = "/home/blitzkrieg/Downloads/spring/grid1/31UGR_20180418T104021_50_007153_6_167685/"

image = rasterio.open(f"{scene_path}/31UGR_20180418T104021_50_007153_6_167685_10m_IR.tif")
mask = rasterio.open(f"{scene_path}/31UGR_20180418T104021_50_007153_6_167685_labels.tif")


fig, axs = plt.subplots(1, 2)

img = image.read(1)
label = mask.read(1)
import numpy as np
print(np.unique(label))

axs[0].imshow(img, cmap="viridis")
axs[1].imshow(label, cmap="viridis")
plt.tight_layout()
plt.show()




