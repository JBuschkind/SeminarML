"""
Training Pipeline for HoVer-Net
Handles training loop, validation, checkpointing, and logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
import json
from typing import Dict, Optional
from tqdm import tqdm
import numpy as np

# Optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

from ..models import HoVerNet, HoVerNetLoss


class Trainer:
    """
    Trainer for HoVer-Net model.
    """
    
    def __init__(
        self,
        model: HoVerNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: HoVerNetLoss,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        save_dir: str = 'outputs/checkpoints',
        log_dir: str = 'outputs/logs',
        save_best: bool = True,
        save_last: bool = True,
        log_interval: int = 10,
        val_interval: int = 1
    ):
        """
        Args:
            model: HoVer-Net model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            loss_fn: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            device: Device to use ('cuda' or 'cpu')
            save_dir: Directory to save checkpoints
            log_dir: Directory for TensorBoard logs
            save_best: Whether to save best model
            save_last: Whether to save last model
            log_interval: Log every N batches
            val_interval: Validate every N epochs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.log_dir = Path(log_dir)
        self.save_best = save_best
        self.save_last = save_last
        self.log_interval = log_interval
        self.val_interval = val_interval
        
        # Create directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer (optional)
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        else:
            self.writer = None
            print("Warning: TensorBoard not available. Install with: pip install tensorboard")
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {
            'total': 0.0,
            'nuclear': 0.0,
            'hover': 0.0,
            'type': 0.0
        }
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device)
            nuclear_target = batch['nuclear'].to(self.device)
            hover_target = batch['hover'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images)
            
            # Prepare targets
            targets = {
                'nuclear': nuclear_target,
                'hover': hover_target
            }
            
            # Compute loss
            losses = self.loss_fn(predictions, targets)
            loss = losses['total_loss']
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Accumulate losses
            epoch_losses['total'] += loss.item()
            epoch_losses['nuclear'] += losses['nuclear_loss'].item()
            epoch_losses['hover'] += losses['hover_loss'].item()
            if 'type_loss' in losses:
                epoch_losses['type'] += losses['type_loss'].item()
            
            num_batches += 1
            self.global_step += 1
            
            # Logging
            if batch_idx % self.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{current_lr:.6f}"
                })
                
                # TensorBoard logging
                if self.writer is not None:
                    self.writer.add_scalar('Train/BatchLoss', loss.item(), self.global_step)
                    self.writer.add_scalar('Train/LearningRate', current_lr, self.global_step)
                    self.writer.add_scalar('Train/NuclearLoss', losses['nuclear_loss'].item(), self.global_step)
                    self.writer.add_scalar('Train/HoverLoss', losses['hover_loss'].item(), self.global_step)
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        val_losses = {
            'total': 0.0,
            'nuclear': 0.0,
            'hover': 0.0,
            'type': 0.0
        }
        num_batches = 0
        
        pbar = tqdm(self.val_loader, desc='Validation')
        
        for batch in pbar:
            # Move to device
            images = batch['image'].to(self.device)
            nuclear_target = batch['nuclear'].to(self.device)
            hover_target = batch['hover'].to(self.device)
            
            # Forward pass
            predictions = self.model(images)
            
            # Prepare targets
            targets = {
                'nuclear': nuclear_target,
                'hover': hover_target
            }
            
            # Compute loss
            losses = self.loss_fn(predictions, targets)
            
            # Accumulate losses
            val_losses['total'] += losses['total_loss'].item()
            val_losses['nuclear'] += losses['nuclear_loss'].item()
            val_losses['hover'] += losses['hover_loss'].item()
            if 'type_loss' in losses:
                val_losses['type'] += losses['type_loss'].item()
            
            num_batches += 1
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= num_batches
        
        return val_losses
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        checkpoint_path = self.save_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, num_epochs: int, resume_from: Optional[str] = None):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from (optional)
        """
        # Resume from checkpoint if provided
        if resume_from:
            self.load_checkpoint(resume_from)
            start_epoch = self.current_epoch + 1
        else:
            start_epoch = 0
        
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Train
            train_losses = self.train_epoch()
            self.train_losses.append(train_losses)
            
            # Validate
            if epoch % self.val_interval == 0:
                val_losses = self.validate()
                self.val_losses.append(val_losses)
                
                # Check if best model
                is_best = val_losses['total'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_losses['total']
                
                # Logging
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                print(f"Train Loss: {train_losses['total']:.4f} "
                      f"(Nuclear: {train_losses['nuclear']:.4f}, "
                      f"HoVer: {train_losses['hover']:.4f})")
                print(f"Val Loss: {val_losses['total']:.4f} "
                      f"(Nuclear: {val_losses['nuclear']:.4f}, "
                      f"HoVer: {val_losses['hover']:.4f})")
                
                # TensorBoard logging
                if self.writer is not None:
                    self.writer.add_scalar('Val/TotalLoss', val_losses['total'], epoch)
                    self.writer.add_scalar('Val/NuclearLoss', val_losses['nuclear'], epoch)
                    self.writer.add_scalar('Val/HoverLoss', val_losses['hover'], epoch)
                
                # Save checkpoint
                if self.save_best and is_best:
                    self.save_checkpoint('best_model.pth', is_best=True)
                    print(f"Saved best model (val_loss: {val_losses['total']:.4f})")
            
            # Save last checkpoint
            if self.save_last:
                self.save_checkpoint('last_model.pth', is_best=False)
            
            # TensorBoard logging for training
            if self.writer is not None:
                self.writer.add_scalar('Train/EpochLoss', train_losses['total'], epoch)
                self.writer.add_scalar('Train/EpochNuclearLoss', train_losses['nuclear'], epoch)
                self.writer.add_scalar('Train/EpochHoverLoss', train_losses['hover'], epoch)
            
            epoch_time = time.time() - start_time
            print(f"Epoch time: {epoch_time:.2f}s\n")
        
        print("Training completed!")
        if self.writer is not None:
            self.writer.close()


if __name__ == '__main__':
    # Test trainer
    print("Trainer module loaded successfully")
