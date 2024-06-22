import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from DataProcess.Dataset import TRANSFORM_IMAGE, SentinelPatchDataset
from model.Resnet50 import CustomResNet50
from model.Unet import CustomUnet
from model.Unet3D import UNet3D


## PARAMS
##! --------------- Model --------------- !##
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_channels = 13  # Multispektral kanal sayısı
num_classes = 9   # Maskedeki sınıf sayısı
MODEL_PATH = "weight/custom01_unet.pth"
patch_size = 64

##! --------------- Dataset --------------- !##
# DATA_PATH = [f"dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/data/Pansharpen/raster/PanComposite_2023-12-01.tif"]
DATA_PATH = [f"dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/data/Resample/raster/CompositeBandsDataset02_2023-12-01.tif"]
MASK_PATH = [f"dataset/ImpactObservatory-LULC_Sentinel2-L1C_10m_Cukurova_v0.0.2/mask/raster/mask.tif"]
old_classes = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
target_classes = torch.tensor([1, 2, 4, 5, 7, 8, 9, 10, 11])



# =================================================================================================================== #
#! Dataset
# =================================================================================================================== #
def ChangeMaskOrder2Old(mask, old_classes, target_classes):
    mapping = {oldc:newc for oldc, newc in zip(old_classes, target_classes)}
    new_mask = mask.clone()
    for oldc, newc in mapping.items():
        new_mask[mask == oldc] = newc
    return new_mask


dataset = SentinelPatchDataset(DATA_PATH, MASK_PATH, patch_size=patch_size)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)



# =================================================================================================================== #
#! Compile Model
# =================================================================================================================== #
# Create Model
# MODEL = CustomResNet50(num_channels, num_classes, patch_size).to(DEVICE)
MODEL = CustomUnet(num_channels, num_classes, patch_size).to(DEVICE).train()
# MODEL = UNet3D(num_channels, num_classes, patch_size).to(DEVICE).train()
MODEL.load_state_dict(torch.load(MODEL_PATH))
MODEL = MODEL.eval()


if "__main__" == __name__:
    inputs, targets = next(iter(dataloader))
    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
    
    inputs = TRANSFORM_IMAGE(inputs)
    outputs = MODEL(inputs)

    # outputs = outputs.squeeze(1) # Unet3D

    class_indices = torch.argmax(outputs, dim=1)
    class_indices = class_indices.unsqueeze(1)
    class_indices = ChangeMaskOrder2Old(class_indices, old_classes, target_classes)

    # Show Patches
    fig, axs = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(16):
        if i<inputs.shape[1]:
            axs[i%4, i//4].imshow(inputs[0, i].cpu().numpy(), cmap="gray")  # Grayscale olarak görselleştirme
        axs[i%4, i//4].axis("off")
    
    axs[2, 3].imshow(targets.cpu().numpy()[0, 0])
    axs[2, 3].text(0, 0, "Ground Truth", fontsize=12, color="green", weight="bold")
    axs[3, 3].imshow(class_indices[0, 0].cpu().numpy())
    axs[3, 3].text(0, 0, "Predicted", fontsize=12, color="blue", weight="bold")
    plt.tight_layout()
    plt.show()