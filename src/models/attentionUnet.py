"""
Attention UNet implementation for brain MRI segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution, ResidualUnit
from monai.networks.layers import Norm
from typing import List, Optional

class AttentionGate(nn.Module):
    """
    Attention gate for skip connections
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi

class AttentionUNet3D(nn.Module):
    """
    3D Attention UNet for brain MRI segmentation
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 3, 
                 features: List[int] = [32, 64, 128, 256], 
                 dropout: float = 0.1, attention_dropout: float = 0.1):
        super(AttentionUNet3D, self).__init__()
        
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
                Convolution(
                    spatial_dims=3,
                    in_channels=features[i],
                    out_channels=features[i + 1],
                    kernel_size=3,
                    padding=1,
                    norm=Norm.BATCH,
                    act="relu",
                    dropout=dropout
                ),
                Convolution(
                    spatial_dims=3,
                    in_channels=features[i + 1],
                    out_channels=features[i + 1],
                    kernel_size=3,
                    padding=1,
                    norm=Norm.BATCH,
                    act="relu",
                    dropout=dropout
                )
            )
            self.encoder_blocks.append(block)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool3d(2),
            Convolution(
                spatial_dims=3,
                in_channels=features[-1],
                out_channels=features[-1] * 2,
                kernel_size=3,
                padding=1,
                norm=Norm.BATCH,
                act="relu",
                dropout=dropout
            ),
            Convolution(
                spatial_dims=3,
                in_channels=features[-1] * 2,
                out_channels=features[-1] * 2,
                kernel_size=3,
                padding=1,
                norm=Norm.BATCH,
                act="relu",
                dropout=dropout
            )
        )
        
        # Attention gates
        self.attention_gates = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            gate = AttentionGate(
                F_g=features[i] * 2 if i == len(features) - 1 else features[i],
                F_l=features[i-1],
                F_int=features[i-1] // 2
            )
            self.attention_gates.append(gate)
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            block = nn.Sequential(
                nn.ConvTranspose3d(
                    features[i] * 2 if i == len(features) - 1 else features[i],
                    features[i-1],
                    kernel_size=2,
                    stride=2
                ),
                Convolution(
                    spatial_dims=3,
                    in_channels=features[i-1] * 2,
                    out_channels=features[i-1],
                    kernel_size=3,
                    padding=1,
                    norm=Norm.BATCH,
                    act="relu",
                    dropout=dropout
                ),
                Convolution(
                    spatial_dims=3,
                    in_channels=features[i-1],
                    out_channels=features[i-1],
                    kernel_size=3,
                    padding=1,
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
        
        # Decoder with attention gates
        for i, (attention_gate, decoder_block) in enumerate(zip(self.attention_gates, self.decoder_blocks)):
            # Get skip connection
            skip_idx = len(self.encoder_outputs) - 1 - i
            skip_x = self.encoder_outputs[skip_idx]
            
            # Apply attention gate
            attention_x = attention_gate(x, skip_x)
            
            # Upsample
            x = decoder_block[0](x)
            
            # Handle size mismatch
            if x.shape != attention_x.shape:
                x = F.interpolate(x, size=attention_x.shape[2:], mode='trilinear', align_corners=True)
            
            # Concatenate
            x = torch.cat([x, attention_x], dim=1)
            
            # Apply decoder convolutions
            x = decoder_block[1](x)
            x = decoder_block[2](x)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x

class AttentionUNet2D(nn.Module):
    """
    2D Attention UNet for slice-wise processing
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 3, 
                 features: List[int] = [32, 64, 128, 256], 
                 dropout: float = 0.1, attention_dropout: float = 0.1):
        super(AttentionUNet2D, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        
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
                Convolution(
                    spatial_dims=2,
                    in_channels=features[i],
                    out_channels=features[i + 1],
                    kernel_size=3,
                    padding=1,
                    norm=Norm.BATCH,
                    act="relu",
                    dropout=dropout
                ),
                Convolution(
                    spatial_dims=2,
                    in_channels=features[i + 1],
                    out_channels=features[i + 1],
                    kernel_size=3,
                    padding=1,
                    norm=Norm.BATCH,
                    act="relu",
                    dropout=dropout
                )
            )
            self.encoder_blocks.append(block)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            Convolution(
                spatial_dims=2,
                in_channels=features[-1],
                out_channels=features[-1] * 2,
                kernel_size=3,
                padding=1,
                norm=Norm.BATCH,
                act="relu",
                dropout=dropout
            ),
            Convolution(
                spatial_dims=2,
                in_channels=features[-1] * 2,
                out_channels=features[-1] * 2,
                kernel_size=3,
                padding=1,
                norm=Norm.BATCH,
                act="relu",
                dropout=dropout
            )
        )
        
        # Attention gates
        self.attention_gates = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            gate = AttentionGate2D(
                F_g=features[i] * 2 if i == len(features) - 1 else features[i],
                F_l=features[i-1],
                F_int=features[i-1] // 2
            )
            self.attention_gates.append(gate)
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            block = nn.Sequential(
                nn.ConvTranspose2d(
                    features[i] * 2 if i == len(features) - 1 else features[i],
                    features[i-1],
                    kernel_size=2,
                    stride=2
                ),
                Convolution(
                    spatial_dims=2,
                    in_channels=features[i-1] * 2,
                    out_channels=features[i-1],
                    kernel_size=3,
                    padding=1,
                    norm=Norm.BATCH,
                    act="relu",
                    dropout=dropout
                ),
                Convolution(
                    spatial_dims=2,
                    in_channels=features[i-1],
                    out_channels=features[i-1],
                    kernel_size=3,
                    padding=1,
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
            
            # Decoder with attention gates
            for i, (attention_gate, decoder_block) in enumerate(zip(self.attention_gates, self.decoder_blocks)):
                # Get skip connection
                skip_idx = len(self.encoder_outputs) - 1 - i
                skip_x = self.encoder_outputs[skip_idx]
                
                # Apply attention gate
                attention_x = attention_gate(slice_x, skip_x)
                
                # Upsample
                slice_x = decoder_block[0](slice_x)
                
                # Handle size mismatch
                if slice_x.shape != attention_x.shape:
                    slice_x = F.interpolate(slice_x, size=attention_x.shape[2:], mode='bilinear', align_corners=True)
                
                # Concatenate
                slice_x = torch.cat([slice_x, attention_x], dim=1)
                
                # Apply decoder convolutions
                slice_x = decoder_block[1](slice_x)
                slice_x = decoder_block[2](slice_x)
            
            # Final convolution
            slice_x = self.final_conv(slice_x)
            outputs.append(slice_x)
        
        # Stack outputs back to 3D
        output = torch.stack(outputs, dim=2)  # [B, C, D, H, W]
        return output

class AttentionGate2D(nn.Module):
    """
    2D Attention gate for skip connections
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super(AttentionGate2D, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi

def create_attentionunet(model_name: str = "attentionunet3d", **kwargs) -> nn.Module:
    """Create Attention UNet model"""
    if model_name.lower() == "attentionunet3d":
        return AttentionUNet3D(**kwargs)
    elif model_name.lower() == "attentionunet2d":
        return AttentionUNet2D(**kwargs)
    else:
        raise ValueError(f"Unknown Attention UNet variant: {model_name}")

# Example usage
if __name__ == "__main__":
    # Test different variants
    variants = ["attentionunet3d", "attentionunet2d"]
    
    for variant in variants:
        model = create_attentionunet(variant, in_channels=4, out_channels=3)
        x = torch.randn(1, 4, 128, 128, 128)
        output = model(x)
        print(f"Attention UNet {variant} input shape: {x.shape}")
        print(f"Attention UNet {variant} output shape: {output.shape}")
        print(f"Attention UNet {variant} parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("-" * 50)
