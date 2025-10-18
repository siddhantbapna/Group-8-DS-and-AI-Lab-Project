"""
3D UNet implementation for brain MRI segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution, ResidualUnit
from monai.networks.layers import Norm
from typing import List, Tuple, Optional

class DoubleConv(nn.Module):
    """Double convolution block"""
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle different spatial dimensions
        diff_x = x2.size()[2] - x1.size()[2]
        diff_y = x2.size()[3] - x1.size()[3]
        diff_z = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diff_z // 2, diff_z - diff_z // 2,
                        diff_y // 2, diff_y - diff_y // 2,
                        diff_x // 2, diff_x - diff_x // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Final output convolution"""
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    """
    3D UNet for brain MRI segmentation
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 3, 
                 features: List[int] = [32, 64, 128, 256], bilinear: bool = True):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        factor = 2 if bilinear else 1
        self.down4 = Down(features[3], features[3] // factor)
        
        self.up1 = Up(features[3], features[2] // factor, bilinear)
        self.up2 = Up(features[2], features[1] // factor, bilinear)
        self.up3 = Up(features[1], features[0] // factor, bilinear)
        self.up4 = Up(features[0], features[0], bilinear)
        self.outc = OutConv(features[0], out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNet(nn.Module):
    """
    2D UNet for brain MRI segmentation (slice-wise processing)
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 3, 
                 features: List[int] = [32, 64, 128, 256], bilinear: bool = True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv2D(in_channels, features[0])
        self.down1 = Down2D(features[0], features[1])
        self.down2 = Down2D(features[1], features[2])
        self.down3 = Down2D(features[2], features[3])
        factor = 2 if bilinear else 1
        self.down4 = Down2D(features[3], features[3] // factor)
        
        self.up1 = Up2D(features[3], features[2] // factor, bilinear)
        self.up2 = Up2D(features[2], features[1] // factor, bilinear)
        self.up3 = Up2D(features[1], features[0] // factor, bilinear)
        self.up4 = Up2D(features[0], features[0], bilinear)
        self.outc = OutConv2D(features[0], out_channels)

    def forward(self, x):
        # Process each slice independently
        batch_size, channels, depth, height, width = x.shape
        outputs = []
        
        for d in range(depth):
            slice_x = x[:, :, d, :, :]  # [B, C, H, W]
            x1 = self.inc(slice_x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x_slice = self.up1(x5, x4)
            x_slice = self.up2(x_slice, x3)
            x_slice = self.up3(x_slice, x2)
            x_slice = self.up4(x_slice, x1)
            logits = self.outc(x_slice)
            outputs.append(logits)
        
        # Stack outputs back to 3D
        output = torch.stack(outputs, dim=2)  # [B, C, D, H, W]
        return output

class DoubleConv2D(nn.Module):
    """2D Double convolution block"""
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down2D(nn.Module):
    """2D Downscaling with maxpool then double conv"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv2D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up2D(nn.Module):
    """2D Upscaling then double conv"""
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv2D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv2D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle different spatial dimensions
        diff_x = x2.size()[2] - x1.size()[2]
        diff_y = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diff_y // 2, diff_y - diff_y // 2,
                        diff_x // 2, diff_x - diff_x // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv2D(nn.Module):
    """2D Final output convolution"""
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

def create_unet(model_name: str = "unet3d", **kwargs) -> nn.Module:
    """Create UNet model"""
    if model_name.lower() == "unet3d":
        return UNet3D(**kwargs)
    elif model_name.lower() == "unet":
        return UNet(**kwargs)
    else:
        raise ValueError(f"Unknown UNet variant: {model_name}")

# Example usage
if __name__ == "__main__":
    # Test 3D UNet
    model_3d = UNet3D(in_channels=4, out_channels=3)
    x = torch.randn(1, 4, 128, 128, 128)
    output = model_3d(x)
    print(f"3D UNet input shape: {x.shape}")
    print(f"3D UNet output shape: {output.shape}")
    
    # Test 2D UNet
    model_2d = UNet(in_channels=4, out_channels=3)
    x = torch.randn(1, 4, 64, 128, 128)
    output = model_2d(x)
    print(f"2D UNet input shape: {x.shape}")
    print(f"2D UNet output shape: {output.shape}")
