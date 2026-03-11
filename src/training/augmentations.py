"""
Data Augmentation for HoVer-Net Training
Augmentations are applied to both images and masks simultaneously
"""

import numpy as np
import cv2
import random
from typing import Dict
import torch


class AugmentationPipeline:
    """
    Pipeline for data augmentation.
    All augmentations are applied to images and masks together.
    """
    
    def __init__(
        self,
        horizontal_flip: bool = True,
        vertical_flip: bool = True,
        rotation: bool = True,
        rotation_range: tuple = (-15, 15),
        color_jitter: bool = True,
        elastic_deformation: bool = False,
        scale: bool = True,
        scale_range: tuple = (0.9, 1.1)
    ):
        """
        Args:
            horizontal_flip: Whether to apply random horizontal flip
            vertical_flip: Whether to apply random vertical flip
            rotation: Whether to apply random rotation
            rotation_range: Rotation range in degrees (min, max)
            color_jitter: Whether to apply color jittering
            elastic_deformation: Whether to apply elastic deformation (slow)
            scale: Whether to apply random scaling
            scale_range: Scaling range (min, max)
        """
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation = rotation
        self.rotation_range = rotation_range
        self.color_jitter = color_jitter
        self.elastic_deformation = elastic_deformation
        self.scale = scale
        self.scale_range = scale_range
    
    def __call__(self, data: Dict) -> Dict:
        """
        Apply augmentations to data.
        
        Args:
            data: Dictionary with 'image', 'nuclear', 'instance', 'hover'
            
        Returns:
            Augmented data dictionary
        """
        image = data['image'].copy()
        nuclear = data['nuclear'].copy()
        instance = data['instance'].copy()
        hover = data.get('hover', None)
        if hover is not None:
            hover = hover.copy()
        
        # Random horizontal flip
        if self.horizontal_flip and random.random() > 0.5:
            image, nuclear, instance, hover = self._horizontal_flip(
                image, nuclear, instance, hover
            )
        
        # Random vertical flip
        if self.vertical_flip and random.random() > 0.5:
            image, nuclear, instance, hover = self._vertical_flip(
                image, nuclear, instance, hover
            )
        
        # Random rotation
        if self.rotation and random.random() > 0.5:
            angle = random.uniform(*self.rotation_range)
            image, nuclear, instance, hover = self._rotate(
                image, nuclear, instance, hover, angle
            )
        
        # Random scaling
        if self.scale and random.random() > 0.5:
            scale_factor = random.uniform(*self.scale_range)
            image, nuclear, instance, hover = self._scale(
                image, nuclear, instance, hover, scale_factor
            )
        
        # Color jitter
        if self.color_jitter and random.random() > 0.5:
            image = self._color_jitter(image)
        
        # Elastic deformation (optional, slow)
        if self.elastic_deformation and random.random() > 0.3:
            image, nuclear, instance, hover = self._elastic_deformation(
                image, nuclear, instance, hover
            )
        
        result = {
            'image': image,
            'nuclear': nuclear,
            'instance': instance
        }
        
        if hover is not None:
            result['hover'] = hover
        
        return result
    
    def _horizontal_flip(self, image, nuclear, instance, hover):
        """Flip horizontally."""
        image = np.fliplr(image)
        nuclear = np.fliplr(nuclear)
        instance = np.fliplr(instance)
        
        if hover is not None:
            hover = np.fliplr(hover)
            hover[:, :, 0] *= -1  # Invert horizontal component
        
        return image, nuclear, instance, hover
    
    def _vertical_flip(self, image, nuclear, instance, hover):
        """Flip vertically."""
        image = np.flipud(image)
        nuclear = np.flipud(nuclear)
        instance = np.flipud(instance)
        
        if hover is not None:
            hover = np.flipud(hover)
            hover[:, :, 1] *= -1  # Invert vertical component
        
        return image, nuclear, instance, hover
    
    def _rotate(self, image, nuclear, instance, hover, angle):
        """Rotate by angle degrees."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate image
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Rotate masks
        nuclear = cv2.warpAffine(nuclear, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        instance = cv2.warpAffine(instance.astype(np.float32), M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(instance.dtype)
        
        if hover is not None:
            # Rotate HoVer maps (need to rotate vectors)
            hover_h = cv2.warpAffine(hover[:, :, 0], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            hover_v = cv2.warpAffine(hover[:, :, 1], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            # Rotate vectors by angle
            angle_rad = np.deg2rad(angle)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            
            hover_new_h = hover_h * cos_a - hover_v * sin_a
            hover_new_v = hover_h * sin_a + hover_v * cos_a
            
            hover = np.stack([hover_new_h, hover_new_v], axis=-1)
        
        return image, nuclear, instance, hover
    
    def _scale(self, image, nuclear, instance, hover, scale_factor):
        """Scale by factor."""
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        # Resize
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        nuclear = cv2.resize(nuclear, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        instance = cv2.resize(instance.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(instance.dtype)
        
        if hover is not None:
            hover = cv2.resize(hover, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Crop or pad to original size
        if scale_factor > 1.0:
            # Crop center
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            image = image[start_h:start_h+h, start_w:start_w+w]
            nuclear = nuclear[start_h:start_h+h, start_w:start_w+w]
            instance = instance[start_h:start_h+h, start_w:start_w+w]
            if hover is not None:
                hover = hover[start_h:start_h+h, start_w:start_w+w]
        else:
            # Pad
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            image = np.pad(image, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w), (0, 0)), mode='reflect')
            nuclear = np.pad(nuclear, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w)), mode='constant', constant_values=0)
            instance = np.pad(instance, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w)), mode='constant', constant_values=0)
            if hover is not None:
                hover = np.pad(hover, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w), (0, 0)), mode='constant', constant_values=0)
        
        return image, nuclear, instance, hover
    
    def _color_jitter(self, image):
        """Apply color jittering."""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Random brightness
        brightness = random.uniform(0.8, 1.2)
        hsv[:, :, 2] *= brightness
        
        # Random saturation
        saturation = random.uniform(0.8, 1.2)
        hsv[:, :, 1] *= saturation
        
        # Clip values
        hsv = np.clip(hsv, 0, 255)
        
        # Convert back to RGB
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        return image
    
    def _elastic_deformation(self, image, nuclear, instance, hover):
        """Apply elastic deformation (simplified version)."""
        # This is a simplified version
        # Full implementation would use scipy.ndimage
        # For now, skip to avoid dependency
        return image, nuclear, instance, hover


def get_train_augmentation(**kwargs):
    """Get augmentation pipeline for training."""
    # Default values
    defaults = {
        'horizontal_flip': True,
        'vertical_flip': True,
        'rotation': True,
        'rotation_range': (-15, 15),
        'color_jitter': True,
        'elastic_deformation': False,  # Slow, optional
        'scale': True,
        'scale_range': (0.9, 1.1)
    }
    # Update defaults with provided kwargs
    defaults.update(kwargs)
    return AugmentationPipeline(**defaults)


def get_val_augmentation(**kwargs):
    """Get augmentation pipeline for validation (minimal or none)."""
    # Default values for validation (minimal augmentation)
    defaults = {
        'horizontal_flip': False,
        'vertical_flip': False,
        'rotation': False,
        'rotation_range': (-15, 15),  # Not used if rotation=False
        'color_jitter': False,
        'elastic_deformation': False,
        'scale': False,
        'scale_range': (0.9, 1.1)  # Not used if scale=False
    }
    # Update defaults with provided kwargs
    defaults.update(kwargs)
    return AugmentationPipeline(**defaults)


if __name__ == '__main__':
    # Test augmentations
    print("Testing augmentations...")
    
    # Create dummy data
    data = {
        'image': np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
        'nuclear': np.random.randint(0, 2, (256, 256), dtype=np.uint8),
        'instance': np.random.randint(0, 5, (256, 256), dtype=np.int32),
        'hover': np.random.randn(256, 256, 2).astype(np.float32)
    }
    
    aug = get_train_augmentation()
    augmented = aug(data)
    
    print(f"Original image shape: {data['image'].shape}")
    print(f"Augmented image shape: {augmented['image'].shape}")
    print("Augmentations work!")
