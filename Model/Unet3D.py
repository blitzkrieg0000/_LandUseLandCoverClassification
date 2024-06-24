from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from Model.Base import ModelMeta

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# B2-B3-B4-B8          => 10m
# B5-B6-B7-B8A-B11-B12 => 20m
#! B1-B9-B10           => 60m
# LULC için toplam 10 Band

class UNet3D(nn.Module):
    def __init__(self, num_input_channel=1, num_classes=9, patch_size=64):
        super(UNet3D, self).__init__()
        
        # Metadata
        self.__ModelMeta = ModelMeta(
            NumInputChannel=num_input_channel,
            NumClasses=num_classes,
            PatchSize=patch_size
        )

        # Down-sampling işlemleri
        self.down_conv1 = self.conv_block(1, 16)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.down_conv2 = self.conv_block(16, 32)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.down_conv3 = self.conv_block(32, 64)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.bottleneck = self.conv_block(64, 128)
        
        # Up-sampling işlemleri
        self.up_conv1 = self.conv_transpose_block(128, 64)
        self.decoder_conv1 = self.conv_block(128, 64)
        self.up_conv2 = self.conv_transpose_block(64, 32)
        self.decoder_conv2 = self.conv_block(64, 32)
        self.up_conv3 = self.conv_transpose_block(32, 16)
        self.decoder_conv3 = self.conv_block(32, 1)

        # Output katmanı
        self.final_conv = nn.Conv3d(1, 1, kernel_size=1, stride=1, padding=0)


    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
    
    def conv_transpose_block(self, in_channels, out_channels):
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        # =================================================================================================================== #
        #! Encoder
        # =================================================================================================================== #
        x1 = self.down_conv1(x)     # [1, 1, 10, 64, 64]   =>   [1, 16, 10, 64, 64]
        x2 = self.maxpool1(x1)      # [1, 16, 10, 64, 64]   =>   [1, 16, 5, 32, 32]
        
        x3 = self.down_conv2(x2)    # [1, 16, 5, 32, 32]    =>   [1, 32, 5, 32, 32]
        x4 = self.maxpool2(x3)      # [1, 32, 5, 32, 32]   =>   [1, 32, 2, 16, 16]

        x5 = self.down_conv3(x4)    # [1, 32, 2, 16, 16]   =>   [1, 64, 2, 16, 16]
        x6 = self.maxpool3(x5)      # [1, 64, 2, 16, 16]   =>   [1, 64, 1, 8, 8]

        x7 = self.bottleneck(x6)    # [1, 64, 1, 8, 8]     =>   [1, 128, 1, 8, 8]

        # =================================================================================================================== #
        #! Decoder
        # =================================================================================================================== #
        x = self.up_conv1(x7)           # [1, 128, 1, 8, 8] => [1, 64, 2, 16, 16]
        x = torch.cat([x, x5], dim=1)   # [1, 64, 2, 16, 16] |+| [1, 64, 2, 16, 16] => [1, 128, 2, 16, 16]
        
        # Decoder Conv3D - 1
        x = self.decoder_conv1(x)       # [1, 128, 2, 16, 16] => [1, 64, 2, 16, 16]

        x = self.up_conv2(x)            # [1, 64, 2, 16, 16] => [1, 32, 4, 32, 32]
        x = F.interpolate(x, size=(x3.size(2), x3.size(3), x3.size(4)), mode="trilinear", align_corners=True)
        x = torch.cat([x, x3], dim=1)   # [1, 32, 5, 32, 32] |+| [1, 32, 5, 32, 32] => [1, 64, 5, 32, 32]

        # Decoder Conv3D - 2
        x = self.decoder_conv2(x)       # [1, 64, 5, 32, 32] => [1, 32, 5, 32, 32]

        x = self.up_conv3(x)            # [1, 32, 5, 32, 32] => [1, 16, 10, 64, 64]
        x = torch.cat([x, x1], dim=1)   # [1, 16, 10, 64, 64] |+| [1, 16, 10, 64, 64] => [1, 32, 10, 64, 64]

        # Decoder Conv3D - 3
        x = self.decoder_conv3(x)       # [1, 32, 10, 64, 64] => [1, 1, 10, 64, 64]

        # Final convolutional layer
        x = F.interpolate(x, size=(self.__ModelMeta.NumClasses, self.__ModelMeta.PatchSize, self.__ModelMeta.PatchSize), mode="trilinear", align_corners=True)
        x = self.final_conv(x) # => [1, 1, 9, 64, 64]
        return x

    def Metadata(self):
        self.__ModelMeta.InputShape = (self.__ModelMeta.NumInputChannel, self.__ModelMeta.InputDepth, self.__ModelMeta.InputHeight, self.__ModelMeta.InputWidth)
        self.__ModelMeta.OutputShape = (1, self.__ModelMeta.NumClasses, self.__ModelMeta.InputHeight, self.__ModelMeta.InputWidth)
        return self.__ModelMeta


if "__main__" == __name__:
    # Modeli oluştur
    model = UNet3D().to(DEVICE)
    summary(model, device=DEVICE)
    # Giriş ve çıkış boyutlarını test et
    input_tensor = torch.randn(1, 1, 10, 64, 64).to(DEVICE)  # (batch_size, channels, depth, height, width)
    output_tensor = model(input_tensor)
    print("Çıkış boyutu:", output_tensor.shape)  # Çıkış boyutu: torch.Size([1, 1, 9, 64, 64])