"""
HoVer-Net Architecture
Encoder-Decoder architecture with Multi-Task Learning for nucleus instance segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Optional, Tuple


class ResNetEncoder(nn.Module):
    """
    ResNet-based encoder for feature extraction.
    Uses ResNet34, ResNet50, or ResNeXt50 as backbone.
    """
    
    def __init__(self, backbone: str = 'resnet34', pretrained: bool = True):
        """
        Args:
            backbone: 'resnet34', 'resnet50', or 'resnext50_32x4d'
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        # Load pretrained ResNet
        if backbone == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            self.out_channels = [64, 64, 128, 256, 512]
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.out_channels = [64, 256, 512, 1024, 2048]
        elif backbone == 'resnext50_32x4d':
            resnet = models.resnext50_32x4d(pretrained=pretrained)
            self.out_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Extract layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # stride=1, output: H/4, W/4
        self.layer2 = resnet.layer2  # stride=2, output: H/8, W/8
        self.layer3 = resnet.layer3  # stride=2, output: H/16, W/16
        self.layer4 = resnet.layer4  # stride=2, output: H/32, W/32
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input image (B, 3, H, W)
            
        Returns:
            Dictionary with multi-scale features:
                - 'x0': Initial features (B, 64, H/4, W/4)
                - 'x1': Layer1 output (B, C1, H/4, W/4)
                - 'x2': Layer2 output (B, C2, H/8, W/8)
                - 'x3': Layer3 output (B, C3, H/16, W/16)
                - 'x4': Layer4 output (B, C4, H/32, W/32)
        """
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # (B, 64, H/4, W/4)
        x0 = x
        
        # ResNet layers
        x1 = self.layer1(x0)   # (B, C1, H/4, W/4)
        x2 = self.layer2(x1)   # (B, C2, H/8, W/8)
        x3 = self.layer3(x2)   # (B, C3, H/16, W/16)
        x4 = self.layer4(x3)   # (B, C4, H/32, W/32)
        
        return {
            'x0': x0,
            'x1': x1,
            'x2': x2,
            'x3': x3,
            'x4': x4
        }


class DecoderBlock(nn.Module):
    """
    Decoder block with skip connection (U-Net style).
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_bn: bool = True
    ):
        """
        Args:
            in_channels: Number of input channels (from previous decoder layer)
            skip_channels: Number of channels from skip connection (encoder)
            out_channels: Number of output channels
            use_bn: Whether to use batch normalization
        """
        super().__init__()
        
        # Upsampling
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2
        )
        
        # Convolution after concatenation
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input from previous decoder layer (B, in_channels, H, W)
            skip: Skip connection from encoder (B, skip_channels, H, W)
            
        Returns:
            Output (B, out_channels, H*2, W*2)
        """
        # Upsample
        x = self.upsample(x)
        
        # Ensure same size (in case of size mismatch)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Convolution
        x = self.conv(x)
        
        return x


class HoVerNetDecoder(nn.Module):
    """
    U-Net style decoder with skip connections.
    """
    
    def __init__(
        self,
        encoder_channels: list,
        decoder_channels: int = 256
    ):
        """
        Args:
            encoder_channels: List of encoder output channels [C0, C1, C2, C3, C4]
            decoder_channels: Number of channels in decoder
        """
        super().__init__()
        
        # Bottom layer (starts from x4)
        self.bottom = nn.Sequential(
            nn.Conv2d(encoder_channels[4], decoder_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True)
        )
        
        # Decoder blocks (bottom-up)
        self.decoder4 = DecoderBlock(decoder_channels, encoder_channels[3], decoder_channels)
        self.decoder3 = DecoderBlock(decoder_channels, encoder_channels[2], decoder_channels)
        self.decoder2 = DecoderBlock(decoder_channels, encoder_channels[1], decoder_channels)
        self.decoder1 = DecoderBlock(decoder_channels, encoder_channels[0], decoder_channels)
        
        # Final upsampling to original resolution
        self.final_upsample = nn.ConvTranspose2d(
            decoder_channels, decoder_channels, kernel_size=2, stride=2
        )
        
    def forward(self, encoder_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            encoder_features: Dictionary with 'x0', 'x1', 'x2', 'x3', 'x4'
            
        Returns:
            Decoder output (B, decoder_channels, H, W)
        """
        # Start from bottom
        x = self.bottom(encoder_features['x4'])
        
        # Decoder blocks with skip connections
        x = self.decoder4(x, encoder_features['x3'])
        x = self.decoder3(x, encoder_features['x2'])
        x = self.decoder2(x, encoder_features['x1'])
        x = self.decoder1(x, encoder_features['x0'])
        
        # Final upsampling
        x = self.final_upsample(x)
        
        return x


class HoVerNet(nn.Module):
    """
    HoVer-Net: Multi-task learning for nucleus instance segmentation.
    
    Architecture:
    - Encoder: ResNet backbone
    - Decoder: U-Net style with skip connections
    - Multi-Task Heads: Nuclear segmentation, HoVer maps, Type classification
    """
    
    def __init__(
        self,
        backbone: str = 'resnet34',
        pretrained: bool = True,
        num_types: Optional[int] = None,
        decoder_channels: int = 256
    ):
        """
        Args:
            backbone: 'resnet34', 'resnet50', or 'resnext50_32x4d'
            pretrained: Whether to use pretrained encoder weights
            num_types: Number of nucleus types (None = no type classification)
            decoder_channels: Number of channels in decoder
        """
        super().__init__()
        
        # Encoder
        self.encoder = ResNetEncoder(backbone=backbone, pretrained=pretrained)
        encoder_channels = self.encoder.out_channels
        
        # Decoder
        self.decoder = HoVerNetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels
        )
        
        # Multi-Task Heads
        # 1. Nuclear Segmentation Head
        self.nuclear_head = nn.Sequential(
            nn.Conv2d(decoder_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
            # No Sigmoid - will use BCEWithLogitsLoss (better for Mixed Precision)
        )
        
        # 2. HoVer Map Head
        self.hover_head = nn.Sequential(
            nn.Conv2d(decoder_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1),
            nn.Tanh()  # Output [-1, 1]
        )
        
        # 3. Type Classification Head (optional)
        self.num_types = num_types
        if num_types is not None and num_types > 0:
            self.type_head = nn.Sequential(
                nn.Conv2d(decoder_channels, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, num_types, kernel_size=1)
                # No activation - will use CrossEntropyLoss
            )
        else:
            self.type_head = None
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input images (B, 3, H, W)
            
        Returns:
            Dictionary with:
                - 'nuclear': Nuclear segmentation logits (B, 1, H, W) - apply sigmoid to get [0, 1]
                - 'hover': HoVer maps (B, 2, H, W) [-1, 1]
                - 'type': Type classification (B, num_types, H, W) if num_types > 0
        """
        # Handle variable input sizes
        original_size = x.shape[2:]
        
        # Encoder
        encoder_features = self.encoder(x)
        
        # Decoder
        decoder_output = self.decoder(encoder_features)
        
        # Ensure output matches input size
        if decoder_output.shape[2:] != original_size:
            decoder_output = F.interpolate(
                decoder_output,
                size=original_size,
                mode='bilinear',
                align_corners=False
            )
        
        # Multi-Task Heads
        nuclear_pred = self.nuclear_head(decoder_output)
        hover_pred = self.hover_head(decoder_output)
        
        result = {
            'nuclear': nuclear_pred,
            'hover': hover_pred
        }
        
        # Type classification (optional)
        if self.type_head is not None:
            type_pred = self.type_head(decoder_output)
            result['type'] = type_pred
        
        return result
    
    def predict_instances(
        self,
        nuclear_pred: torch.Tensor,
        hover_pred: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Post-process predictions to get instance segmentation.
        Uses watershed algorithm (simplified version).
        
        Args:
            nuclear_pred: Nuclear segmentation prediction (B, 1, H, W)
            hover_pred: HoVer map prediction (B, 2, H, W)
            threshold: Threshold for nuclear segmentation
            
        Returns:
            Instance map (B, H, W) with unique IDs for each instance
        """
        # This is a simplified version
        # Full implementation would use watershed algorithm
        nuclear_binary = (nuclear_pred > threshold).squeeze(1)  # (B, H, W)
        
        # TODO: Implement proper watershed using hover_pred
        # For now, return binary mask
        return nuclear_binary.long()


if __name__ == '__main__':
    # Test the model
    model = HoVerNet(
        backbone='resnet34',
        pretrained=True,
        num_types=4,  # 4 nucleus types
        decoder_channels=256
    )
    
    # Test input
    x = torch.randn(2, 3, 512, 512)
    
    # Forward pass
    outputs = model(x)
    
    print("Model outputs:")
    print(f"  Nuclear: {outputs['nuclear'].shape}")
    print(f"  HoVer: {outputs['hover'].shape}")
    print(f"  Type: {outputs['type'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
