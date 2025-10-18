"""
nnUNet implementation for brain MRI segmentation
Based on the nnU-Net framework architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution, ResidualUnit
from monai.networks.layers import Norm
from typing import List, Optional

class nnUNet3D(nn.Module):
    """
    3D nnUNet implementation for brain MRI segmentation
    Based on the nnU-Net framework
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 3, 
                 features: List[int] = [32, 64, 128, 256, 320, 320],
                 dropout: float = 0.1):
        super(nnUNet3D, self).__init__()
        
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
            act="relu"
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
                    kernel_size=3,
                    subunits=2,
                    norm=Norm.BATCH,
                    act="relu",
                    dropout=dropout
                )
            )
            self.encoder_blocks.append(block)
        
        # Bottleneck
        self.bottleneck = ResidualUnit(
            spatial_dims=3,
            in_channels=features[-1],
            out_channels=features[-1],
            kernel_size=3,
            subunits=2,
            norm=Norm.BATCH,
            act="relu",
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
                    kernel_size=3,
                    subunits=2,
                    norm=Norm.BATCH,
                    act="relu",
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

class nnUNet3D_Deep(nn.Module):
    """
    Deep 3D nnUNet with more layers
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 3, 
                 features: List[int] = [32, 64, 128, 256, 320, 320, 320],
                 dropout: float = 0.1):
        super(nnUNet3D_Deep, self).__init__()
        
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
            act="relu"
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
                    kernel_size=3,
                    subunits=2,
                    norm=Norm.BATCH,
                    act="relu",
                    dropout=dropout
                )
            )
            self.encoder_blocks.append(block)
        
        # Bottleneck
        self.bottleneck = ResidualUnit(
            spatial_dims=3,
            in_channels=features[-1],
            out_channels=features[-1],
            kernel_size=3,
            subunits=2,
            norm=Norm.BATCH,
            act="relu",
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
                    kernel_size=3,
                    subunits=2,
                    norm=Norm.BATCH,
                    act="relu",
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

class nnUNet3D_Light(nn.Module):
    """
    Lightweight 3D nnUNet
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 3, 
                 features: List[int] = [16, 32, 64, 128, 256],
                 dropout: float = 0.1):
        super(nnUNet3D_Light, self).__init__()
        
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
            act="relu"
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
                    kernel_size=3,
                    subunits=1,
                    norm=Norm.BATCH,
                    act="relu",
                    dropout=dropout
                )
            )
            self.encoder_blocks.append(block)
        
        # Bottleneck
        self.bottleneck = ResidualUnit(
            spatial_dims=3,
            in_channels=features[-1],
            out_channels=features[-1],
            kernel_size=3,
            subunits=1,
            norm=Norm.BATCH,
            act="relu",
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
                    kernel_size=3,
                    subunits=1,
                    norm=Norm.BATCH,
                    act="relu",
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

def create_nnunet(model_name: str = "nnunet3d", **kwargs) -> nn.Module:
    """Create nnUNet model"""
    if model_name.lower() == "nnunet3d":
        return nnUNet3D(**kwargs)
    elif model_name.lower() == "nnunet3d_deep":
        return nnUNet3D_Deep(**kwargs)
    elif model_name.lower() == "nnunet3d_light":
        return nnUNet3D_Light(**kwargs)
    else:
        raise ValueError(f"Unknown nnUNet variant: {model_name}")

# Example usage
if __name__ == "__main__":
    # Test different variants
    variants = ["nnunet3d", "nnunet3d_deep", "nnunet3d_light"]
    
    for variant in variants:
        model = create_nnunet(variant, in_channels=4, out_channels=3)
        x = torch.randn(1, 4, 128, 128, 128)
        output = model(x)
        print(f"nnUNet {variant} input shape: {x.shape}")
        print(f"nnUNet {variant} output shape: {output.shape}")
        print(f"nnUNet {variant} parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("-" * 50)
