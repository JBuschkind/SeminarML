"""
Training Script for HoVer-Net
"""

import sys
import argparse
import yaml
from pathlib import Path
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import HoVerNet, HoVerNetLoss
from src.data.dataloader import get_dataloaders
from src.training.trainer import Trainer
from src.training.augmentations import get_train_augmentation, get_val_augmentation


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_optimizer(model, config: dict):
    """Create optimizer from config."""
    optimizer_name = config['training']['optimizer'].lower()
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer, config: dict):
    """Create learning rate scheduler from config."""
    scheduler_type = config['training']['scheduler']
    if scheduler_type is None:
        return None
    
    scheduler_type = scheduler_type.lower()
    params = config['training']['scheduler_params']
    
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=params['T_max'])
    elif scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=params['step_size'], gamma=params['gamma'])
    else:
        return None
    
    return scheduler


def main():
    parser = argparse.ArgumentParser(description='Train HoVer-Net')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    
    # Device
    device = config.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Data loaders
    print("\nCreating data loaders...")
    train_aug = get_train_augmentation(**config['augmentation']['train'])
    val_aug = get_val_augmentation(**config['augmentation']['val'])
    
    dataloaders = get_dataloaders(
        data_dir=config['data']['data_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        split_file=config['data']['split_file'],
        transform_train=train_aug,
        transform_val=val_aug,
        cache_masks=config['data']['cache_masks'],
        generate_hover=config['data']['generate_hover']
    )
    
    print(f"Train batches: {len(dataloaders['train'])}")
    print(f"Val batches: {len(dataloaders['val'])}")
    
    # Model
    print("\nCreating model...")
    model = HoVerNet(
        backbone=config['model']['backbone'],
        pretrained=config['model']['pretrained'],
        num_types=config['model']['num_types'],
        decoder_channels=config['model']['decoder_channels']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss function
    print("\nCreating loss function...")
    loss_fn = HoVerNetLoss(
        nuclear_weight=config['loss']['nuclear_weight'],
        hover_weight=config['loss']['hover_weight'],
        type_weight=config['loss']['type_weight'],
        hover_loss_type=config['loss']['hover_loss_type'],
        use_dice=config['loss']['use_dice']
    )
    
    # Optimizer
    print("\nCreating optimizer...")
    optimizer = create_optimizer(model, config)
    
    # Scheduler
    scheduler = create_scheduler(optimizer, config)
    if scheduler:
        print(f"Using {config['training']['scheduler']} scheduler")
    
    # Trainer
    print("\nCreating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=config['output']['save_dir'],
        log_dir=config['output']['log_dir'],
        save_best=config['output']['save_best'],
        save_last=config['output']['save_last'],
        log_interval=config['output']['log_interval'],
        val_interval=config['output']['val_interval']
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    trainer.train(
        num_epochs=config['training']['num_epochs'],
        resume_from=args.resume
    )
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()
