"""
Mask Generator for HoVer-Net
Converts polygon annotations to:
- Nuclear Segmentation Maps (binary mask)
- Instance Maps (each cell has unique ID)
- HoVer Maps (Horizontal/Vertical vectors for instance separation)
"""

import numpy as np
from skimage.draw import polygon
from skimage.morphology import binary_dilation, disk
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import cdist
from typing import Dict, Tuple, Optional
import cv2


def generate_masks(
    regions: list,
    image_shape: Tuple[int, int],
    generate_hover: bool = True,
    generate_type: bool = False
) -> Dict[str, np.ndarray]:
    """
    Generate masks from polygon regions.
    
    Args:
        regions: List of region dictionaries with 'vertices' (Nx2 array)
        image_shape: (height, width) of the output masks
        generate_hover: Whether to generate HoVer maps
        generate_type: Whether to generate type classification map
        
    Returns:
        Dictionary with:
            - 'nuclear': Binary nuclear segmentation map (H, W)
            - 'instance': Instance map with unique IDs (H, W)
            - 'hover': HoVer maps (H, W, 2) - [horizontal, vertical]
            - 'type': Type classification map (H, W) - optional
    """
    height, width = image_shape
    
    # Initialize masks
    nuclear_mask = np.zeros((height, width), dtype=np.uint8)
    instance_mask = np.zeros((height, width), dtype=np.int32)
    hover_map = np.zeros((height, width, 2), dtype=np.float32) if generate_hover else None
    
    # Process each region
    for idx, region in enumerate(regions):
        instance_id = idx + 1
        vertices = region['vertices'].astype(int)
        
        # Ensure vertices are within image bounds
        vertices[:, 0] = np.clip(vertices[:, 0], 0, width - 1)
        vertices[:, 1] = np.clip(vertices[:, 1], 0, height - 1)
        
        # Fill polygon
        if len(vertices) >= 3:
            rr, cc = polygon(vertices[:, 1], vertices[:, 0], shape=(height, width))
            nuclear_mask[rr, cc] = 1
            instance_mask[rr, cc] = instance_id
    
    # Generate HoVer maps if requested
    if generate_hover and hover_map is not None:
        hover_map = generate_hover_maps(instance_mask, nuclear_mask)
    
    result = {
        'nuclear': nuclear_mask,
        'instance': instance_mask,
    }
    
    if hover_map is not None:
        result['hover'] = hover_map
    
    return result


def generate_hover_maps(
    instance_mask: np.ndarray,
    nuclear_mask: np.ndarray
) -> np.ndarray:
    """
    Generate Horizontal/Vertical maps for instance separation.
    
    Uses optimized distance transform approach for better performance.
    
    Args:
        instance_mask: Instance map with unique IDs (H, W)
        nuclear_mask: Binary nuclear segmentation map (H, W)
        
    Returns:
        HoVer maps (H, W, 2) where:
            - [:, :, 0] = horizontal component
            - [:, :, 1] = vertical component
    """
    return generate_hover_maps_optimized(instance_mask, nuclear_mask)


def generate_hover_maps_optimized(
    instance_mask: np.ndarray,
    nuclear_mask: np.ndarray
) -> np.ndarray:
    """
    Optimized version using distance transform and gradient.
    This is faster and more efficient than the pixel-by-pixel approach.
    """
    height, width = instance_mask.shape
    hover_map = np.zeros((height, width, 2), dtype=np.float32)
    
    # Get unique instance IDs
    instance_ids = np.unique(instance_mask)
    instance_ids = instance_ids[instance_ids > 0]
    
    for instance_id in instance_ids:
        instance_binary = (instance_mask == instance_id).astype(np.uint8)
        
        if np.sum(instance_binary) == 0:
            continue
        
        # Distance transform from boundary (distance from center to boundary)
        # For pixels inside the instance, this gives distance to nearest boundary
        dist_transform = distance_transform_edt(instance_binary)
        
        # Compute gradient to get direction to boundary
        # Use Sobel operators to compute gradients
        dist_float = dist_transform.astype(np.float32)
        sobel_x = cv2.Sobel(dist_float, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(dist_float, cv2.CV_32F, 0, 1, ksize=3)
        
        # Normalize gradients to get unit vectors
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        mask = magnitude > 1e-6  # Avoid division by zero
        sobel_x[mask] /= magnitude[mask]
        sobel_y[mask] /= magnitude[mask]
        
        # Invert direction (point towards boundary, not away from center)
        # The gradient points away from boundary, so we negate it
        instance_mask_bool = instance_binary > 0
        hover_map[instance_mask_bool, 0] = -sobel_x[instance_mask_bool]
        hover_map[instance_mask_bool, 1] = -sobel_y[instance_mask_bool]
    
    return hover_map


if __name__ == '__main__':
    # Test with dummy data
    test_regions = [
        {'vertices': np.array([[10, 10], [20, 10], [20, 20], [10, 20], [10, 10]])},
        {'vertices': np.array([[30, 30], [40, 30], [40, 40], [30, 40], [30, 30]])},
    ]
    
    masks = generate_masks(test_regions, (50, 50))
    print(f"Nuclear mask shape: {masks['nuclear'].shape}")
    print(f"Instance mask shape: {masks['instance'].shape}")
    print(f"HoVer map shape: {masks['hover'].shape}")
    print(f"Unique instances: {np.unique(masks['instance'])}")
