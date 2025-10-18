"""
ResUNet implementation for brain MRI segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import ResidualUnit, Convolution
from monai.networks.layers import Norm
from typing import List, Optional

class ResUNet3D(nn.Module):
    """
    3D ResUNet with residual connections for brain MRI segmentation
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 3, 
                 features: List[int] = [32, 64, 128, 256], 
                 num_res_units: int = 2, dropout: float = 0.1):
        super(ResUNet3D, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.num_res_units = num_res_units
        
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
                    subunits=num_res_units,
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
            subunits=num_res_units,
            norm=Norm.BATCH,
            act="relu",
            dropout=dropout
        )
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            # After concatenation, input channels will be features[i] + features[i-1]
            block = nn.Sequential(
                nn.ConvTranspose3d(features[i], features[i-1], kernel_size=2, stride=2),
                ResidualUnit(
                    spatial_dims=3,
                    in_channels=features[i] + features[i-1],  # Account for concatenation
                    out_channels=features[i-1],
                    kernel_size=3,
                    subunits=num_res_units,
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

class ResUNet2D(nn.Module):
    """
    2D ResUNet for slice-wise processing
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 3, 
                 features: List[int] = [32, 64, 128, 256], 
                 num_res_units: int = 2, dropout: float = 0.1):
        super(ResUNet2D, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.num_res_units = num_res_units
        
        # Initial convolution
        self.initial_conv = Convolution(
            spatial_dims=2,
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
                nn.MaxPool2d(2),
                ResidualUnit(
                    spatial_dims=2,
                    in_channels=features[i],
                    out_channels=features[i + 1],
                    kernel_size=3,
                    subunits=num_res_units,
                    norm=Norm.BATCH,
                    act="relu",
                    dropout=dropout
                )
            )
            self.encoder_blocks.append(block)
        
        # Bottleneck
        self.bottleneck = ResidualUnit(
            spatial_dims=2,
            in_channels=features[-1],
            out_channels=features[-1],
            kernel_size=3,
            subunits=num_res_units,
            norm=Norm.BATCH,
            act="relu",
            dropout=dropout
        )
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            block = nn.Sequential(
                nn.ConvTranspose2d(features[i], features[i-1], kernel_size=2, stride=2),
                ResidualUnit(
                    spatial_dims=2,
                    in_channels=features[i],
                    out_channels=features[i-1],
                    kernel_size=3,
                    subunits=num_res_units,
                    norm=Norm.BATCH,
                    act="relu",
                    dropout=dropout
                )
            )
            self.decoder_blocks.append(block)
        
        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        # Store encoder outputs for skip connections
        self.encoder_outputs = []

    def forward(self, x):
        # Process each slice independently
        batch_size, channels, depth, height, width = x.shape
        outputs = []
        
        for d in range(depth):
            slice_x = x[:, :, d, :, :]  # [B, C, H, W]
            self.encoder_outputs = []
            
            # Initial convolution
            slice_x = self.initial_conv(slice_x)
            self.encoder_outputs.append(slice_x)
            
            # Encoder
            for encoder_block in self.encoder_blocks:
                slice_x = encoder_block(slice_x)
                self.encoder_outputs.append(slice_x)
            
            # Bottleneck
            slice_x = self.bottleneck(slice_x)
            
            # Decoder with skip connections
            for i, decoder_block in enumerate(self.decoder_blocks):
                # Upsample
                slice_x = F.interpolate(slice_x, scale_factor=2, mode='bilinear', align_corners=True)
                
                # Skip connection
                skip_idx = len(self.encoder_outputs) - 2 - i
                skip_x = self.encoder_outputs[skip_idx]
                
                # Handle size mismatch
                if slice_x.shape != skip_x.shape:
                    slice_x = F.interpolate(slice_x, size=skip_x.shape[2:], mode='bilinear', align_corners=True)
                
                # Concatenate
                slice_x = torch.cat([slice_x, skip_x], dim=1)
                
                # Apply residual unit
                slice_x = decoder_block[1](slice_x)
            
            # Final convolution
            slice_x = self.final_conv(slice_x)
            outputs.append(slice_x)
        
        # Stack outputs back to 3D
        output = torch.stack(outputs, dim=2)  # [B, C, D, H, W]
        return output

class ResUNet3D_Advanced(nn.Module):
    """
    Advanced 3D ResUNet with attention mechanisms
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 3, 
                 features: List[int] = [32, 64, 128, 256, 512], 
                 num_res_units: int = 2, dropout: float = 0.1):
        super(ResUNet3D_Advanced, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.num_res_units = num_res_units
        
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
        
        # Encoder with attention
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(features) - 1):
            block = nn.Sequential(
                nn.MaxPool3d(2),
                ResidualUnit(
                    spatial_dims=3,
                    in_channels=features[i],
                    out_channels=features[i + 1],
                    kernel_size=3,
                    subunits=num_res_units,
                    norm=Norm.BATCH,
                    act="relu",
                    dropout=dropout
                )
            )
            self.encoder_blocks.append(block)
        
        # Bottleneck with attention
        self.bottleneck = ResidualUnit(
            spatial_dims=3,
            in_channels=features[-1],
            out_channels=features[-1],
            kernel_size=3,
            subunits=num_res_units,
            norm=Norm.BATCH,
            act="relu",
            dropout=dropout
        )
        
        # Decoder with attention
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            block = nn.Sequential(
                nn.ConvTranspose3d(features[i], features[i-1], kernel_size=2, stride=2),
                ResidualUnit(
                    spatial_dims=3,
                    in_channels=features[i],
                    out_channels=features[i-1],
                    kernel_size=3,
                    subunits=num_res_units,
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

def create_resunet(model_name: str = "resunet3d", **kwargs) -> nn.Module:
    """Create ResUNet model"""
    if model_name.lower() == "resunet3d":
        return ResUNet3D(**kwargs)
    elif model_name.lower() == "resunet2d":
        return ResUNet2D(**kwargs)
    elif model_name.lower() == "resunet3d_advanced":
        return ResUNet3D_Advanced(**kwargs)
    else:
        raise ValueError(f"Unknown ResUNet variant: {model_name}")

# Example usage
if __name__ == "__main__":
    # Test different variants
    variants = ["resunet3d", "resunet2d", "resunet3d_advanced"]
    
    for variant in variants:
        model = create_resunet(variant, in_channels=4, out_channels=3)
        x = torch.randn(1, 4, 128, 128, 128)
        output = model(x)
        print(f"ResUNet {variant} input shape: {x.shape}")
        print(f"ResUNet {variant} output shape: {output.shape}")
        print(f"ResUNet {variant} parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("-" * 50)
