"""
PyTorch Dataset for HoVer-Net
Loads images and corresponding masks for training
"""

import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import cv2

from .xml_parser import parse_xml_annotations
from .mask_generator import generate_masks


class NucleusDataset(Dataset):
    """
    Dataset for nucleus segmentation with HoVer-Net.
    
    Loads:
    - Images (TIF format)
    - Nuclear segmentation masks
    - Instance maps
    - HoVer maps
    """
    
    def __init__(
        self,
        data_dir: str,
        image_list: List[str],
        transform: Optional[callable] = None,
        cache_masks: bool = False,
        generate_hover: bool = True
    ):
        """
        Args:
            data_dir: Root directory containing the dataset
            image_list: List of image filenames (without extension) to use
            transform: Optional transform to apply to images and masks
            cache_masks: Whether to cache generated masks in memory
            generate_hover: Whether to generate HoVer maps
        """
        self.data_dir = Path(data_dir)
        self.image_list = image_list
        self.transform = transform
        self.cache_masks = cache_masks
        self.generate_hover = generate_hover
        
        # Cache for masks
        self.mask_cache = {} if cache_masks else None
        
        # Find corresponding sample directories
        self.samples = []
        for img_name in image_list:
            # Find the sample directory containing this image
            img_name_no_ext = img_name.replace('.tif', '').replace('.xml', '')
            
            # Search for matching files
            for sample_dir in self.data_dir.iterdir():
                if not sample_dir.is_dir():
                    continue
                
                # Check for matching TIF and XML files
                tif_file = sample_dir / f"{img_name_no_ext}.tif"
                xml_file = sample_dir / f"{img_name_no_ext}.xml"
                
                if tif_file.exists() and xml_file.exists():
                    self.samples.append({
                        'image_path': str(tif_file),
                        'annotation_path': str(xml_file),
                        'name': img_name_no_ext
                    })
                    break
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            Dictionary with:
                - 'image': Image tensor (C, H, W) - RGB
                - 'nuclear': Nuclear segmentation mask (H, W)
                - 'instance': Instance map (H, W)
                - 'hover': HoVer maps (H, W, 2) - optional
                - 'name': Sample name
        """
        sample = self.samples[idx]
        
        # Load image
        image = self._load_image(sample['image_path'])
        original_shape = image.shape[:2]
        
        # Load or generate masks
        if self.cache_masks and idx in self.mask_cache:
            masks = self.mask_cache[idx]
        else:
            masks = self._load_masks(sample['annotation_path'], original_shape)
            if self.cache_masks:
                self.mask_cache[idx] = masks
        
        # Apply transforms if provided
        if self.transform:
            # Stack image and masks for joint transformation
            data = {
                'image': image,
                **masks
            }
            data = self.transform(data)
            image = data['image']
            masks = {k: v for k, v in data.items() if k != 'image'}
        
        # Helper function to ensure array is contiguous
        def ensure_contiguous(arr):
            """Ensure array is contiguous (no negative strides)."""
            if not arr.flags['C_CONTIGUOUS']:
                return arr.copy()
            return arr
        
        # Convert to tensors
        # Image: (H, W, C) -> (C, H, W)
        if len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))
        image = ensure_contiguous(image)
        image = torch.from_numpy(image).float()
        
        # Normalize image to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        
        # Convert masks to tensors
        result = {
            'image': image,
            'name': sample['name']
        }
        
        # Nuclear mask
        nuclear = ensure_contiguous(masks['nuclear'])
        nuclear = torch.from_numpy(nuclear).long()
        result['nuclear'] = nuclear
        
        # Instance map
        instance = ensure_contiguous(masks['instance'])
        instance = torch.from_numpy(instance).long()
        result['instance'] = instance
        
        # HoVer maps
        if 'hover' in masks and masks['hover'] is not None:
            hover = masks['hover']
            # Convert from (H, W, 2) to (2, H, W)
            hover = np.transpose(hover, (2, 0, 1))
            hover = ensure_contiguous(hover)
            hover = torch.from_numpy(hover).float()
            result['hover'] = hover
        
        return result
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image from file."""
        image = Image.open(image_path)
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)
    
    def _load_masks(
        self,
        annotation_path: str,
        image_shape: Tuple[int, int]
    ) -> Dict[str, np.ndarray]:
        """Load and generate masks from XML annotation."""
        # Parse XML
        annotations = parse_xml_annotations(annotation_path)
        regions = annotations['regions']
        
        # Generate masks
        masks = generate_masks(
            regions,
            image_shape,
            generate_hover=self.generate_hover
        )
        
        return masks


def pad_to_size(tensor: torch.Tensor, target_size: Tuple[int, int], pad_value: float = 0.0) -> torch.Tensor:
    """
    Pad tensor to target size.
    
    Args:
        tensor: Tensor to pad (C, H, W) or (H, W)
        target_size: Target (height, width)
        pad_value: Value to use for padding
        
    Returns:
        Padded tensor
    """
    if len(tensor.shape) == 3:
        C, H, W = tensor.shape
        target_h, target_w = target_size
    elif len(tensor.shape) == 2:
        H, W = tensor.shape
        target_h, target_w = target_size
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")
    
    # Calculate padding needed
    pad_h = max(0, target_h - H)
    pad_w = max(0, target_w - W)
    
    if pad_h == 0 and pad_w == 0:
        return tensor
    
    # Apply padding: F.pad expects (pad_left, pad_right, pad_top, pad_bottom)
    # For 3D tensors (C, H, W): pad last two dimensions (H, W)
    # For 2D tensors (H, W): pad both dimensions
    if len(tensor.shape) == 3:
        # Pad (H, W) dimensions: (pad_left, pad_right, pad_top, pad_bottom)
        padding = (0, pad_w, 0, pad_h)
    else:
        # Pad (H, W) dimensions: (pad_left, pad_right, pad_top, pad_bottom)
        padding = (0, pad_w, 0, pad_h)
    
    return F.pad(tensor, padding, mode='constant', value=pad_value)


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle variable-sized images.
    Pads all images and masks to the maximum size in the batch.
    """
    # Find maximum dimensions in the batch
    max_h = max(item['image'].shape[1] for item in batch)
    max_w = max(item['image'].shape[2] for item in batch)
    target_size = (max_h, max_w)
    
    # Pad and stack images
    padded_images = []
    padded_nuclear = []
    padded_instance = []
    padded_hover = [] if 'hover' in batch[0] else None
    
    for item in batch:
        # Pad image
        img = item['image']
        if img.shape[1] != max_h or img.shape[2] != max_w:
            img = pad_to_size(img, target_size, pad_value=0.0)
        padded_images.append(img)
        
        # Pad nuclear mask
        nuc = item['nuclear']
        if nuc.shape[0] != max_h or nuc.shape[1] != max_w:
            nuc = pad_to_size(nuc, target_size, pad_value=0)
        padded_nuclear.append(nuc)
        
        # Pad instance mask
        inst = item['instance']
        if inst.shape[0] != max_h or inst.shape[1] != max_w:
            inst = pad_to_size(inst, target_size, pad_value=0)
        padded_instance.append(inst)
        
        # Pad HoVer maps if present
        if padded_hover is not None and 'hover' in item:
            hover = item['hover']
            if hover.shape[1] != max_h or hover.shape[2] != max_w:
                hover = pad_to_size(hover, target_size, pad_value=0.0)
            padded_hover.append(hover)
    
    # Stack all tensors
    result = {
        'image': torch.stack(padded_images),
        'nuclear': torch.stack(padded_nuclear),
        'instance': torch.stack(padded_instance),
        'name': [item['name'] for item in batch]
    }
    
    # Stack HoVer maps if present
    if padded_hover is not None:
        result['hover'] = torch.stack(padded_hover)
    
    return result


if __name__ == '__main__':
    # Test the dataset
    data_dir = r"D:\Git\SeminarML\diy_impl\training_data"
    
    # Get a sample image list
    sample_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if sample_dirs:
        sample_dir = os.path.join(data_dir, sample_dirs[0])
        files = os.listdir(sample_dir)
        tif_files = [f.replace('.tif', '') for f in files if f.endswith('.tif')]
        
        if tif_files:
            dataset = NucleusDataset(data_dir, tif_files[:2])
            print(f"Dataset size: {len(dataset)}")
            
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"Image shape: {sample['image'].shape}")
                print(f"Nuclear mask shape: {sample['nuclear'].shape}")
                print(f"Instance mask shape: {sample['instance'].shape}")
                if 'hover' in sample:
                    print(f"HoVer map shape: {sample['hover'].shape}")
