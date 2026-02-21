"""
Test script for the Dataloader
Tests data loading, mask generation, and visualization
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataloader import get_dataloaders, create_train_val_test_split
from src.evaluation.visualizer import visualize_annotations, visualize_with_hover
import matplotlib.pyplot as plt


def test_dataloader():
    """Test the dataloader functionality."""
    print("=" * 60)
    print("Testing HoVer-Net Dataloader")
    print("=" * 60)
    
    # Paths
    data_dir = project_root / "training_data"
    split_file = project_root / "data" / "splits" / "train_val_test_split.json"
    
    # Create output directory
    output_dir = project_root / "outputs" / "test_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n1. Creating train/val/test split...")
    splits = create_train_val_test_split(
        str(data_dir),
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        save_path=str(split_file)
    )
    
    print(f"\n2. Creating DataLoaders...")
    dataloaders = get_dataloaders(
        str(data_dir),
        batch_size=2,
        num_workers=0,  # Set to 0 for debugging on Windows
        split_file=str(split_file),
        cache_masks=False,
        generate_hover=True
    )
    
    print(f"\n3. Testing DataLoaders...")
    for split_name, loader in dataloaders.items():
        print(f"\n{split_name.upper()} Loader:")
        print(f"  Dataset size: {len(loader.dataset)}")
        print(f"  Number of batches: {len(loader)}")
        
        if len(loader) > 0:
            # Get a batch
            batch = next(iter(loader))
            print(f"  Batch keys: {list(batch.keys())}")
            print(f"  Image shape: {batch['image'].shape}")
            print(f"  Nuclear mask shape: {batch['nuclear'].shape}")
            print(f"  Instance mask shape: {batch['instance'].shape}")
            if 'hover' in batch:
                print(f"  HoVer map shape: {batch['hover'].shape}")
            
            # Visualize first sample in batch
            print(f"\n  Visualizing first sample...")
            image = batch['image'][0].numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
            masks = {
                'nuclear': batch['nuclear'][0].numpy(),
                'instance': batch['instance'][0].numpy(),
            }
            
            if 'hover' in batch:
                hover = batch['hover'][0].numpy()
                # Convert from (2, H, W) to (H, W, 2)
                hover = hover.transpose(1, 2, 0)
                masks['hover'] = hover
            
            # Save visualization
            vis_path = output_dir / f"{split_name}_sample_0.png"
            visualize_annotations(
                image,
                masks,
                save_path=str(vis_path),
                show=False
            )
            
            # Save HoVer visualization if available
            if 'hover' in masks:
                hover_vis_path = output_dir / f"{split_name}_hover_sample_0.png"
                visualize_with_hover(
                    image,
                    masks,
                    save_path=str(hover_vis_path),
                    show=False
                )
    
    print(f"\n4. Test completed!")
    print(f"   Visualizations saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    test_dataloader()
