"""
V-Net implementation for brain MRI segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution, ResidualUnit
from monai.networks.layers import Norm
from typing import List, Optional

class VNet3D(nn.Module):
    """
    3D V-Net for brain MRI segmentation
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 3, 
                 features: List[int] = [16, 32, 64, 128, 256],
                 act: str = "relu", norm: str = "batch", dropout: float = 0.1):
        super(VNet3D, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        
        # Initial convolution
        self.initial_conv = Convolution(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=features[0],
            kernel_size=5,
            padding=2,
            norm=Norm.BATCH,
            act=act
        )
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(features) - 1):
            block = nn.Sequential(
                nn.MaxPool3d(2),
                Convolution(
                    spatial_dims=3,
                    in_channels=features[i],
                    out_channels=features[i + 1],
                    kernel_size=5,
                    padding=2,
                    norm=Norm.BATCH,
                    act=act,
                    dropout=dropout
                )
            )
            self.encoder_blocks.append(block)
        
        # Bottleneck
        self.bottleneck = Convolution(
            spatial_dims=3,
            in_channels=features[-1],
            out_channels=features[-1],
            kernel_size=5,
            padding=2,
            norm=Norm.BATCH,
            act=act,
            dropout=dropout
        )
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            block = nn.Sequential(
                nn.ConvTranspose3d(features[i], features[i-1], kernel_size=2, stride=2),
                Convolution(
                    spatial_dims=3,
                    in_channels=features[i],
                    out_channels=features[i-1],
                    kernel_size=5,
                    padding=2,
                    norm=Norm.BATCH,
                    act=act,
                    dropout=dropout
                )
            )
            self.decoder_blocks.append(block)
        
        # Final convolution
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
        
        # Store encoder outputs for skip connections
        self.encoder_outputs = []

    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)
        self.encoder_outputs = [x]
        
        # Encoder
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            self.encoder_outputs.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        for i, decoder_block in enumerate(self.decoder_blocks):
            # Upsample
            x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
            
            # Skip connection
            skip_idx = len(self.encoder_outputs) - 2 - i
            skip_x = self.encoder_outputs[skip_idx]
            
            # Handle size mismatch
            if x.shape != skip_x.shape:
                x = F.interpolate(x, size=skip_x.shape[2:], mode='trilinear', align_corners=True)
            
            # Concatenate
            x = torch.cat([x, skip_x], dim=1)
            
            # Apply convolution
            x = decoder_block[1](x)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x

class VNet3D_Residual(nn.Module):
    """
    3D V-Net with residual connections
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 3, 
                 features: List[int] = [16, 32, 64, 128, 256],
                 act: str = "relu", norm: str = "batch", dropout: float = 0.1):
        super(VNet3D_Residual, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        
        # Initial convolution
        self.initial_conv = Convolution(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=features[0],
            kernel_size=5,
            padding=2,
            norm=Norm.BATCH,
            act=act
        )
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(features) - 1):
            block = nn.Sequential(
                nn.MaxPool3d(2),
                ResidualUnit(
                    spatial_dims=3,
                    in_channels=features[i],
                    out_channels=features[i + 1],
                    kernel_size=5,
                    subunits=2,
                    norm=Norm.BATCH,
                    act=act,
                    dropout=dropout
                )
            )
            self.encoder_blocks.append(block)
        
        # Bottleneck
        self.bottleneck = ResidualUnit(
            spatial_dims=3,
            in_channels=features[-1],
            out_channels=features[-1],
            kernel_size=5,
            subunits=2,
            norm=Norm.BATCH,
            act=act,
            dropout=dropout
        )
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            block = nn.Sequential(
                nn.ConvTranspose3d(features[i], features[i-1], kernel_size=2, stride=2),
                ResidualUnit(
                    spatial_dims=3,
                    in_channels=features[i],
                    out_channels=features[i-1],
                    kernel_size=5,
                    subunits=2,
                    norm=Norm.BATCH,
                    act=act,
                    dropout=dropout
                )
            )
            self.decoder_blocks.append(block)
        
        # Final convolution
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
        
        # Store encoder outputs for skip connections
        self.encoder_outputs = []

    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)
        self.encoder_outputs = [x]
        
        # Encoder
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            self.encoder_outputs.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        for i, decoder_block in enumerate(self.decoder_blocks):
            # Upsample
            x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
            
            # Skip connection
            skip_idx = len(self.encoder_outputs) - 2 - i
            skip_x = self.encoder_outputs[skip_idx]
            
            # Handle size mismatch
            if x.shape != skip_x.shape:
                x = F.interpolate(x, size=skip_x.shape[2:], mode='trilinear', align_corners=True)
            
            # Concatenate
            x = torch.cat([x, skip_x], dim=1)
            
            # Apply residual unit
            x = decoder_block[1](x)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x

class VNet3D_Light(nn.Module):
    """
    Lightweight 3D V-Net
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 3, 
                 features: List[int] = [8, 16, 32, 64, 128],
                 act: str = "relu", norm: str = "batch", dropout: float = 0.1):
        super(VNet3D_Light, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        
        # Initial convolution
        self.initial_conv = Convolution(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=features[0],
            kernel_size=3,
            padding=1,
            norm=Norm.BATCH,
            act=act
        )
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(features) - 1):
            block = nn.Sequential(
                nn.MaxPool3d(2),
                Convolution(
                    spatial_dims=3,
                    in_channels=features[i],
                    out_channels=features[i + 1],
                    kernel_size=3,
                    padding=1,
                    norm=Norm.BATCH,
                    act=act,
                    dropout=dropout
                )
            )
            self.encoder_blocks.append(block)
        
        # Bottleneck
        self.bottleneck = Convolution(
            spatial_dims=3,
            in_channels=features[-1],
            out_channels=features[-1],
            kernel_size=3,
            padding=1,
            norm=Norm.BATCH,
            act=act,
            dropout=dropout
        )
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            block = nn.Sequential(
                nn.ConvTranspose3d(features[i], features[i-1], kernel_size=2, stride=2),
                Convolution(
                    spatial_dims=3,
                    in_channels=features[i],
                    out_channels=features[i-1],
                    kernel_size=3,
                    padding=1,
                    norm=Norm.BATCH,
                    act=act,
                    dropout=dropout
                )
            )
            self.decoder_blocks.append(block)
        
        # Final convolution
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
        
        # Store encoder outputs for skip connections
        self.encoder_outputs = []

    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)
        self.encoder_outputs = [x]
        
        # Encoder
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            self.encoder_outputs.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        for i, decoder_block in enumerate(self.decoder_blocks):
            # Upsample
            x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
            
            # Skip connection
            skip_idx = len(self.encoder_outputs) - 2 - i
            skip_x = self.encoder_outputs[skip_idx]
            
            # Handle size mismatch
            if x.shape != skip_x.shape:
                x = F.interpolate(x, size=skip_x.shape[2:], mode='trilinear', align_corners=True)
            
            # Concatenate
            x = torch.cat([x, skip_x], dim=1)
            
            # Apply convolution
            x = decoder_block[1](x)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x

def create_vnet(model_name: str = "vnet3d", **kwargs) -> nn.Module:
    """Create V-Net model"""
    if model_name.lower() == "vnet3d":
        return VNet3D(**kwargs)
    elif model_name.lower() == "vnet3d_residual":
        return VNet3D_Residual(**kwargs)
    elif model_name.lower() == "vnet3d_light":
        return VNet3D_Light(**kwargs)
    else:
        raise ValueError(f"Unknown V-Net variant: {model_name}")

# Example usage
if __name__ == "__main__":
    # Test different variants
    variants = ["vnet3d", "vnet3d_residual", "vnet3d_light"]
    
    for variant in variants:
        model = create_vnet(variant, in_channels=4, out_channels=3)
        x = torch.randn(1, 4, 128, 128, 128)
        output = model(x)
        print(f"V-Net {variant} input shape: {x.shape}")
        print(f"V-Net {variant} output shape: {output.shape}")
        print(f"V-Net {variant} parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("-" * 50)
