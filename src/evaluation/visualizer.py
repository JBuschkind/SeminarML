"""
Visualization tools for annotations and predictions
Creates visualizations similar to the example image with colored outlines
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Dict, Optional, Tuple
from pathlib import Path


def visualize_annotations(
    image: np.ndarray,
    masks: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    alpha: float = 0.6,
    show: bool = True
) -> np.ndarray:
    """
    Visualize annotations on image with colored outlines.
    
    Similar to the example image:
    - Blue outlines: Cell nuclei (Nuclear Segmentation)
    - Green outlines: Cell boundaries/interstitium (optional)
    - Red points: Other cellular structures (optional)
    
    Args:
        image: Input image (H, W, 3) RGB
        masks: Dictionary with 'nuclear', 'instance', optionally 'hover'
        save_path: Optional path to save the visualization
        alpha: Transparency for overlays
        show: Whether to display the image
        
    Returns:
        Visualization image (H, W, 3) RGB
    """
    # Create a copy of the image
    vis_image = image.copy()
    
    # Convert to float for blending
    if vis_image.dtype == np.uint8:
        vis_image = vis_image.astype(np.float32) / 255.0
    
    # Get nuclear mask
    nuclear_mask = masks.get('nuclear', None)
    instance_mask = masks.get('instance', None)
    
    if nuclear_mask is None and instance_mask is not None:
        nuclear_mask = (instance_mask > 0).astype(np.uint8)
    
    # Draw blue outlines for nuclei
    if nuclear_mask is not None:
        # Find contours
        contours, _ = cv2.findContours(
            nuclear_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Draw blue outlines
        cv2.drawContours(vis_image, contours, -1, (0, 0, 1.0), 2)  # Blue in RGB
    
    # Draw instance boundaries in different colors if instance mask is available
    if instance_mask is not None:
        # Get unique instance IDs
        instance_ids = np.unique(instance_mask)
        instance_ids = instance_ids[instance_ids > 0]
        
        # Draw each instance with a slightly different shade
        for instance_id in instance_ids:
            instance_binary = (instance_mask == instance_id).astype(np.uint8)
            contours, _ = cv2.findContours(
                instance_binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Use blue for nuclei (main color)
            cv2.drawContours(vis_image, contours, -1, (0, 0, 1.0), 1)
    
    # Convert back to uint8
    vis_image = np.clip(vis_image * 255, 0, 255).astype(np.uint8)
    
    # Create figure
    if show or save_path:
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(vis_image)
        ax.axis('off')
        ax.set_title('Annotated Image - Blue: Cell Nuclei', fontsize=14)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Visualization saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    return vis_image


def visualize_predictions(
    image: np.ndarray,
    ground_truth: Dict[str, np.ndarray],
    prediction: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    show: bool = True
) -> np.ndarray:
    """
    Visualize predictions compared to ground truth.
    
    Args:
        image: Input image (H, W, 3) RGB
        ground_truth: Ground truth masks dictionary
        prediction: Prediction masks dictionary
        save_path: Optional path to save the visualization
        show: Whether to display the image
        
    Returns:
        Visualization image
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth
    gt_vis = visualize_annotations(image, ground_truth, show=False)
    axes[1].imshow(gt_vis)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    pred_vis = visualize_annotations(image, prediction, show=False)
    axes[2].imshow(pred_vis)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Comparison saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return pred_vis


def visualize_with_hover(
    image: np.ndarray,
    masks: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    show: bool = True
) -> np.ndarray:
    """
    Visualize annotations with HoVer map visualization.
    
    Args:
        image: Input image (H, W, 3) RGB
        masks: Dictionary with 'nuclear', 'instance', 'hover'
        save_path: Optional path to save the visualization
        show: Whether to display the image
        
    Returns:
        Visualization image
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Annotated image
    annotated = visualize_annotations(image, masks, show=False)
    axes[0, 1].imshow(annotated)
    axes[0, 1].set_title('Annotated (Blue: Nuclei)')
    axes[0, 1].axis('off')
    
    # HoVer map - Horizontal
    if 'hover' in masks and masks['hover'] is not None:
        hover = masks['hover']
        if len(hover.shape) == 3:
            hover_h = hover[:, :, 0]  # Horizontal component
            hover_v = hover[:, :, 1]  # Vertical component
        else:
            hover_h = hover[0]  # (2, H, W) format
            hover_v = hover[1]
        
        im1 = axes[1, 0].imshow(hover_h, cmap='RdBu', vmin=-1, vmax=1)
        axes[1, 0].set_title('HoVer Map - Horizontal')
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0])
        
        im2 = axes[1, 1].imshow(hover_v, cmap='RdBu', vmin=-1, vmax=1)
        axes[1, 1].set_title('HoVer Map - Vertical')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Visualization saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return annotated


if __name__ == '__main__':
    # Test visualization
    from src.data.dataset import NucleusDataset
    
    data_dir = r"D:\Git\SeminarML\diy_impl\training_data"
    
    # Get a sample
    import os
    sample_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if sample_dirs:
        sample_dir = os.path.join(data_dir, sample_dirs[0])
        files = os.listdir(sample_dir)
        tif_files = [f.replace('.tif', '') for f in files if f.endswith('.tif')]
        
        if tif_files:
            dataset = NucleusDataset(data_dir, tif_files[:1])
            if len(dataset) > 0:
                sample = dataset[0]
                image = sample['image'].numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                masks = {
                    'nuclear': sample['nuclear'].numpy(),
                    'instance': sample['instance'].numpy(),
                    'hover': sample.get('hover', None)
                }
                
                visualize_annotations(image, masks, save_path='test_visualization.png')
