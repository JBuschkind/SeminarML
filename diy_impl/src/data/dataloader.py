"""
DataLoader setup for HoVer-Net
Handles train/val/test splits and creates PyTorch DataLoaders
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader
import torch

from .dataset import NucleusDataset, collate_fn


def create_train_val_test_split(
    data_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    save_path: Optional[str] = None
) -> Dict[str, List[str]]:
    """
    Create train/val/test split on sample level (not image level).
    
    Args:
        data_dir: Root directory containing the dataset
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_seed: Random seed for reproducibility
        save_path: Optional path to save the split information
        
    Returns:
        Dictionary with 'train', 'val', 'test' lists of sample names
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    data_path = Path(data_dir)
    
    # Get all unique sample directories (TCGA-XX-XXXX-...)
    sample_dirs = [d for d in os.listdir(data_path) 
                   if os.path.isdir(data_path / d) and d.startswith('TCGA-')]
    
    # Get all images from each sample
    sample_to_images = {}
    for sample_dir in sample_dirs:
        sample_path = data_path / sample_dir
        tif_files = [f.replace('.tif', '') for f in os.listdir(sample_path) 
                     if f.endswith('.tif')]
        if tif_files:
            sample_to_images[sample_dir] = tif_files
    
    # Shuffle samples
    np.random.seed(random_seed)
    sample_names = list(sample_to_images.keys())
    np.random.shuffle(sample_names)
    
    # Split samples
    n_samples = len(sample_names)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    train_samples = sample_names[:n_train]
    val_samples = sample_names[n_train:n_train + n_val]
    test_samples = sample_names[n_train + n_val:]
    
    # Create image lists for each split
    splits = {
        'train': [],
        'val': [],
        'test': []
    }
    
    for sample_name in train_samples:
        splits['train'].extend(sample_to_images[sample_name])
    
    for sample_name in val_samples:
        splits['val'].extend(sample_to_images[sample_name])
    
    for sample_name in test_samples:
        splits['test'].extend(sample_to_images[sample_name])
    
    # Save split information
    if save_path:
        split_info = {
            'train_samples': train_samples,
            'val_samples': val_samples,
            'test_samples': test_samples,
            'train_images': splits['train'],
            'val_images': splits['val'],
            'test_images': splits['test'],
            'n_train_samples': len(train_samples),
            'n_val_samples': len(val_samples),
            'n_test_samples': len(test_samples),
            'n_train_images': len(splits['train']),
            'n_val_images': len(splits['val']),
            'n_test_images': len(splits['test'])
        }
        
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', 
                   exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"Split saved to {save_path}")
        print(f"Train: {len(train_samples)} samples, {len(splits['train'])} images")
        print(f"Val: {len(val_samples)} samples, {len(splits['val'])} images")
        print(f"Test: {len(test_samples)} samples, {len(splits['test'])} images")
    
    return splits


def load_split_from_file(split_path: str) -> Dict[str, List[str]]:
    """Load split information from JSON file."""
    with open(split_path, 'r') as f:
        split_info = json.load(f)
    
    return {
        'train': split_info['train_images'],
        'val': split_info['val_images'],
        'test': split_info['test_images']
    }


def get_dataloaders(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    split_file: Optional[str] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    transform_train: Optional[callable] = None,
    transform_val: Optional[callable] = None,
    cache_masks: bool = False,
    generate_hover: bool = True,
    pin_memory: bool = True
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        data_dir: Root directory containing the dataset
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        split_file: Optional path to JSON file with split information
        train_ratio: Ratio for training set (if split_file not provided)
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        transform_train: Transform to apply to training data
        transform_val: Transform to apply to validation/test data
        cache_masks: Whether to cache masks in memory
        generate_hover: Whether to generate HoVer maps
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    # Load or create splits
    if split_file and os.path.exists(split_file):
        splits = load_split_from_file(split_file)
    else:
        splits = create_train_val_test_split(
            data_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
    
    # Create datasets
    train_dataset = NucleusDataset(
        data_dir,
        splits['train'],
        transform=transform_train,
        cache_masks=cache_masks,
        generate_hover=generate_hover
    )
    
    val_dataset = NucleusDataset(
        data_dir,
        splits['val'],
        transform=transform_val,
        cache_masks=cache_masks,
        generate_hover=generate_hover
    )
    
    test_dataset = NucleusDataset(
        data_dir,
        splits['test'],
        transform=transform_val,
        cache_masks=cache_masks,
        generate_hover=generate_hover
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


if __name__ == '__main__':
    # Test the dataloader
    data_dir = r"D:\Git\SeminarML\diy_impl\training_data"
    
    # Create splits
    splits = create_train_val_test_split(
        data_dir,
        save_path='data/splits/train_val_test_split.json'
    )
    
    # Create dataloaders
    dataloaders = get_dataloaders(
        data_dir,
        batch_size=2,
        num_workers=0,  # Set to 0 for debugging
        split_file='data/splits/train_val_test_split.json'
    )
    
    # Test loading a batch
    print("\nTesting DataLoader...")
    for split_name, loader in dataloaders.items():
        print(f"\n{split_name.upper()} Loader:")
        print(f"  Dataset size: {len(loader.dataset)}")
        
        if len(loader) > 0:
            batch = next(iter(loader))
            print(f"  Batch keys: {batch.keys()}")
            print(f"  Image shape: {batch['image'].shape}")
            print(f"  Nuclear mask shape: {batch['nuclear'].shape}")
            print(f"  Instance mask shape: {batch['instance'].shape}")
            if 'hover' in batch:
                print(f"  HoVer map shape: {batch['hover'].shape}")
