"""
Test script for HoVer-Net model
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.models.hover_net import HoVerNet
from src.models.losses import HoVerNetLoss

def test_model():
    """Test HoVer-Net model."""
    print("=" * 60)
    print("Testing HoVer-Net Model")
    print("=" * 60)
    
    # Create model
    print("\n1. Creating model...")
    model = HoVerNet(
        backbone='resnet34',
        pretrained=False,  # Set to False for faster testing
        num_types=None,  # No type classification for now
        decoder_channels=256
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    batch_size = 2
    height, width = 256, 256
    
    x = torch.randn(batch_size, 3, height, width)
    print(f"   Input shape: {x.shape}")
    
    model.eval()
    with torch.no_grad():
        outputs = model(x)
    
    print(f"   Nuclear output: {outputs['nuclear'].shape}")
    print(f"   HoVer output: {outputs['hover'].shape}")
    
    # Test loss
    print("\n3. Testing loss function...")
    loss_fn = HoVerNetLoss(
        nuclear_weight=1.0,
        hover_weight=1.0,
        hover_loss_type='l1'
    )
    
    # Create dummy targets
    nuclear_target = torch.randint(0, 2, (batch_size, height, width)).float()
    hover_target = torch.tanh(torch.randn(batch_size, 2, height, width))
    
    targets = {
        'nuclear': nuclear_target,
        'hover': hover_target
    }
    
    losses = loss_fn(outputs, targets)
    print(f"   Nuclear loss: {losses['nuclear_loss'].item():.4f}")
    print(f"   HoVer loss: {losses['hover_loss'].item():.4f}")
    print(f"   Total loss: {losses['total_loss'].item():.4f}")
    
    print("\n4. Test completed successfully!")
    print("=" * 60)

if __name__ == '__main__':
    test_model()
