"""
MONAI-based 3D UNet implementation for brain MRI segmentation
"""
import torch
import torch.nn as nn
from monai.networks.nets import UNet as MONAIUNet
from monai.networks.layers import Norm
from typing import List, Optional

class UNet3D_MONAI(nn.Module):
    """
    3D UNet using MONAI implementation for brain MRI segmentation
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 3, 
                 features: List[int] = [32, 64, 128, 256], 
                 dropout: float = 0.1, norm: str = "batch"):
        super(UNet3D_MONAI, self).__init__()
        
        self.unet = MONAIUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=features,
            strides=[2, 2, 2, 2],
            num_res_units=2,
            norm=Norm.BATCH,
            dropout=dropout,
        )

    def forward(self, x):
        return self.unet(x)

class UNet3D_Advanced(nn.Module):
    """
    Advanced 3D UNet with additional features
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 3, 
                 features: List[int] = [32, 64, 128, 256, 512],
                 dropout: float = 0.1, norm: str = "batch"):
        super(UNet3D_Advanced, self).__init__()
        
        # Use deeper architecture
        self.unet = MONAIUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=features,
            strides=[2, 2, 2, 2, 2],
            num_res_units=2,
            norm=Norm.BATCH,
            dropout=dropout,
            act="relu",
        )

    def forward(self, x):
        return self.unet(x)

class UNet3D_Light(nn.Module):
    """
    Lightweight 3D UNet for faster training
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 3, 
                 features: List[int] = [16, 32, 64, 128],
                 dropout: float = 0.1, norm: str = "batch"):
        super(UNet3D_Light, self).__init__()
        
        self.unet = MONAIUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=features,
            strides=[2, 2, 2],
            num_res_units=1,
            norm=Norm.BATCH,
            dropout=dropout,
        )

    def forward(self, x):
        return self.unet(x)

def create_unet3d(variant: str = "standard", **kwargs) -> nn.Module:
    """Create 3D UNet model with different variants"""
    if variant.lower() == "standard":
        return UNet3D_MONAI(**kwargs)
    elif variant.lower() == "advanced":
        return UNet3D_Advanced(**kwargs)
    elif variant.lower() == "light":
        return UNet3D_Light(**kwargs)
    else:
        raise ValueError(f"Unknown UNet3D variant: {variant}")

# Example usage
if __name__ == "__main__":
    # Test different variants
    variants = ["standard", "advanced", "light"]
    
    for variant in variants:
        model = create_unet3d(variant, in_channels=4, out_channels=3)
        x = torch.randn(1, 4, 128, 128, 128)
        output = model(x)
        print(f"UNet3D {variant} input shape: {x.shape}")
        print(f"UNet3D {variant} output shape: {output.shape}")
        print(f"UNet3D {variant} parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("-" * 50)
