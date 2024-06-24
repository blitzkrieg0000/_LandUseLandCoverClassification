from Dataset.Core import RemoteSensingDatasetManager
from matplotlib import pyplot as plt
from Dataset.Enum import DatasetType


if "__main__" == __name__:
    dataloader = RemoteSensingDatasetManager().GetDataloader(DatasetType.Cukurova_IO_LULC)
    buffer, mask = next(iter(dataloader))
    # Show Patches
    fig, axs = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(16):
        if i<buffer.shape[1]:
            axs[i%4, i//4].imshow(buffer[0, i].cpu().numpy(), cmap="gray")  # Grayscale olarak görselleştirme
        axs[i%4, i//4].axis("off")
    axs[3, 3].imshow(mask[0, 0])
    plt.tight_layout()

    # Show Patches
    fig, axs = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(16):
        if i<buffer.shape[1]:
            axs[i%4, i//4].imshow(buffer[1, i].cpu().numpy(), cmap="gray")  # Grayscale olarak görselleştirme
        axs[i%4, i//4].axis("off")
    axs[3, 3].imshow(mask[1, 0])
    plt.tight_layout()
    plt.show()