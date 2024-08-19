import rasterio
from matplotlib import pyplot as plt
import numpy as np

scene_path = "/home/blitzkrieg/Downloads/spring/grid1/32UND_20180505T103021_52_996058_9_080234/"
image = rasterio.open(f"{scene_path}/32UND_20180505T103021_52_996058_9_080234_10m_RGB.tif")
mask = rasterio.open(f"{scene_path}/32UND_20180505T103021_52_996058_9_080234_labels.tif")


fig, axs = plt.subplots(1, 2)
img = image.read(1)
label = mask.read(1)

print(np.unique(label))

axs[0].imshow(img, cmap="viridis")
axs[1].imshow(label, cmap="viridis")
plt.tight_layout()
plt.show()




