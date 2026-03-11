"""
Loss Functions for HoVer-Net Multi-Task Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    Good for imbalanced datasets.
    """
    
    def __init__(self, smooth: float = 1.0):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, 1, H, W) or (B, H, W) in [0, 1]
            target: Ground truth (B, 1, H, W) or (B, H, W) in {0, 1}
            
        Returns:
            Dice loss (scalar)
        """
        # Flatten
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1).float()
        
        # Dice coefficient
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Loss = 1 - Dice
        return 1.0 - dice.mean()


class CombinedBCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy and Dice Loss.
    Often works better than either alone.
    
    Uses BCEWithLogitsLoss for Mixed Precision compatibility.
    """
    
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        """
        Args:
            bce_weight: Weight for BCE loss
            dice_weight: Weight for Dice loss
        """
        super().__init__()
        # Use BCEWithLogitsLoss for Mixed Precision compatibility
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, 1, H, W) - LOGITS (no sigmoid applied)
            target: Ground truth (B, 1, H, W) or (B, H, W) in {0, 1}
        """
        # Ensure target is in correct format
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()
        
        # BCEWithLogitsLoss works with logits directly
        bce_loss = self.bce(pred, target)
        
        # For Dice loss, we need probabilities [0, 1]
        pred_probs = torch.sigmoid(pred)
        dice_loss = self.dice(pred_probs, target)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class HoVerLoss(nn.Module):
    """
    Loss for HoVer Maps (Horizontal/Vertical vectors).
    Uses L1 or L2 loss.
    """
    
    def __init__(self, loss_type: str = 'l1'):
        """
        Args:
            loss_type: 'l1' or 'l2'
        """
        super().__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='mean')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='mean')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, 2, H, W) in [-1, 1]
            target: Ground truth (B, 2, H, W) in [-1, 1]
            mask: Optional mask to ignore certain pixels (B, H, W) or (B, 1, H, W)
                  Only compute loss where mask > 0
        """
        if mask is not None:
            # Apply mask
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B, 1, H, W)
            mask = mask.float()
            
            # Only compute loss on valid pixels
            pred_masked = pred * mask
            target_masked = target * mask
            
            # Compute loss only on masked regions
            loss = self.loss_fn(pred_masked, target_masked)
            
            # Normalize by number of valid pixels
            num_valid = mask.sum()
            if num_valid > 0:
                loss = loss * (mask.numel() / num_valid)
        else:
            loss = self.loss_fn(pred, target)
        
        return loss


class HoVerNetLoss(nn.Module):
    """
    Combined loss for HoVer-Net multi-task learning.
    Combines:
    - Nuclear segmentation loss (BCE + Dice)
    - HoVer map loss (L1/L2)
    - Type classification loss (CrossEntropy, optional)
    """
    
    def __init__(
        self,
        nuclear_weight: float = 1.0,
        hover_weight: float = 1.0,
        type_weight: float = 0.5,
        hover_loss_type: str = 'l1',
        use_dice: bool = True
    ):
        """
        Args:
            nuclear_weight: Weight for nuclear segmentation loss
            hover_weight: Weight for HoVer map loss
            type_weight: Weight for type classification loss
            hover_loss_type: 'l1' or 'l2' for HoVer loss
            use_dice: Whether to use Dice loss for nuclear segmentation
        """
        super().__init__()
        
        # Nuclear segmentation loss
        if use_dice:
            self.nuclear_loss = CombinedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
        else:
            # Use BCEWithLogitsLoss for Mixed Precision compatibility
            self.nuclear_loss = nn.BCEWithLogitsLoss()
        
        # HoVer map loss
        self.hover_loss = HoVerLoss(loss_type=hover_loss_type)
        
        # Type classification loss (optional)
        self.type_loss = nn.CrossEntropyLoss(ignore_index=-1)
        
        # Weights
        self.nuclear_weight = nuclear_weight
        self.hover_weight = hover_weight
        self.type_weight = type_weight
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            predictions: Dictionary with 'nuclear', 'hover', optionally 'type'
            targets: Dictionary with 'nuclear', 'hover', optionally 'type'
            
        Returns:
            Dictionary with individual losses and total loss:
                - 'nuclear_loss': Nuclear segmentation loss
                - 'hover_loss': HoVer map loss
                - 'type_loss': Type classification loss (if applicable)
                - 'total_loss': Combined loss
        """
        losses = {}
        
        # Nuclear segmentation loss
        nuclear_pred = predictions['nuclear']
        nuclear_target = targets['nuclear']
        
        # Ensure target is in correct format
        if nuclear_target.dim() == 3:
            nuclear_target = nuclear_target.unsqueeze(1).float()
        else:
            nuclear_target = nuclear_target.float()
        
        nuclear_loss = self.nuclear_loss(nuclear_pred, nuclear_target)
        losses['nuclear_loss'] = nuclear_loss
        
        # HoVer map loss
        hover_pred = predictions['hover']
        hover_target = targets['hover']
        
        # Use nuclear mask to only compute loss on nucleus pixels
        nuclear_mask = (nuclear_target > 0.5).float()
        hover_loss = self.hover_loss(hover_pred, hover_target, mask=nuclear_mask)
        losses['hover_loss'] = hover_loss
        
        # Type classification loss (optional)
        if 'type' in predictions and 'type' in targets:
            type_pred = predictions['type']  # (B, num_types, H, W)
            type_target = targets['type']     # (B, H, W) with class indices
            
            # Reshape for CrossEntropy
            B, num_types, H, W = type_pred.shape
            type_pred = type_pred.view(B, num_types, H * W)
            type_target = type_target.view(B, H * W).long()
            
            type_loss = self.type_loss(type_pred, type_target)
            losses['type_loss'] = type_loss
        else:
            losses['type_loss'] = torch.tensor(0.0, device=nuclear_pred.device)
        
        # Total loss
        total_loss = (
            self.nuclear_weight * losses['nuclear_loss'] +
            self.hover_weight * losses['hover_loss'] +
            self.type_weight * losses['type_loss']
        )
        losses['total_loss'] = total_loss
        
        return losses


if __name__ == '__main__':
    # Test losses
    batch_size = 2
    height, width = 256, 256
    
    # Create dummy predictions and targets
    nuclear_pred = torch.sigmoid(torch.randn(batch_size, 1, height, width))
    nuclear_target = torch.randint(0, 2, (batch_size, height, width)).float()
    
    hover_pred = torch.tanh(torch.randn(batch_size, 2, height, width))
    hover_target = torch.tanh(torch.randn(batch_size, 2, height, width))
    
    # Test individual losses
    print("Testing individual losses...")
    
    # Nuclear loss
    nuclear_loss_fn = CombinedBCEDiceLoss()
    nuclear_loss = nuclear_loss_fn(nuclear_pred, nuclear_target.unsqueeze(1))
    print(f"Nuclear loss: {nuclear_loss.item():.4f}")
    
    # HoVer loss
    hover_loss_fn = HoVerLoss(loss_type='l1')
    hover_loss = hover_loss_fn(hover_pred, hover_target)
    print(f"HoVer loss: {hover_loss.item():.4f}")
    
    # Combined loss
    print("\nTesting combined loss...")
    predictions = {
        'nuclear': nuclear_pred,
        'hover': hover_pred
    }
    targets = {
        'nuclear': nuclear_target,
        'hover': hover_target
    }
    
    combined_loss_fn = HoVerNetLoss()
    losses = combined_loss_fn(predictions, targets)
    
    print(f"Nuclear loss: {losses['nuclear_loss'].item():.4f}")
    print(f"HoVer loss: {losses['hover_loss'].item():.4f}")
    print(f"Total loss: {losses['total_loss'].item():.4f}")
