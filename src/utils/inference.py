"""
Inference Pipeline for HoVer-Net
Processes new images and generates predictions
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Union, Dict, Optional, Tuple
import cv2

from ..models import HoVerNet
from ..evaluation.metrics import get_instance_map_from_predictions


class HoVerNetInference:
    """
    Inference class for HoVer-Net model.
    Handles loading models, processing images, and generating predictions.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        threshold: float = 0.5
    ):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to use ('cuda' or 'cpu')
            threshold: Threshold for nuclear segmentation
        """
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.threshold = threshold
        
        # Load model from checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Determine model architecture from checkpoint or config
        # Try to infer from checkpoint
        if 'model_state_dict' in checkpoint:
            # Create model (need to know architecture)
            # For now, use default - could be improved to save config in checkpoint
            self.model = HoVerNet(
                backbone='resnet34',
                pretrained=False,
                num_types=None,
                decoder_channels=256
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise ValueError("Checkpoint does not contain model_state_dict")
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model from {checkpoint_path}")
        print(f"Using device: {self.device}")
    
    def preprocess_image(self, image: Union[np.ndarray, Image.Image, str]) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (numpy array, PIL Image, or path to image)
            
        Returns:
            Preprocessed image tensor (1, 3, H, W) normalized to [0, 1]
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = np.array(image)
        elif isinstance(image, Image.Image):
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = np.array(image)
        
        # Ensure numpy array
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)
        
        # Convert to tensor: (H, W, C) -> (C, H, W)
        if len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))
        
        # Add batch dimension: (C, H, W) -> (1, C, H, W)
        image = torch.from_numpy(image).unsqueeze(0)
        
        return image.to(self.device)
    
    def predict(
        self,
        image: Union[np.ndarray, Image.Image, str],
        return_instances: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Predict on a single image.
        
        Args:
            image: Input image
            return_instances: Whether to return instance map
            
        Returns:
            Dictionary with:
                - 'nuclear': Nuclear segmentation (H, W) in [0, 1]
                - 'hover': HoVer maps (H, W, 2) in [-1, 1]
                - 'instance': Instance map (H, W) with unique IDs (if return_instances=True)
                - 'nuclear_binary': Binary nuclear mask (H, W) in {0, 1}
        """
        # Preprocess
        image_tensor = self.preprocess_image(image)
        original_shape = image_tensor.shape[2:]
        
        # Inference
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Extract predictions
        nuclear_pred = torch.sigmoid(predictions['nuclear']).cpu().numpy()
        hover_pred = predictions['hover'].cpu().numpy()
        
        # Remove batch dimension
        nuclear_pred = nuclear_pred[0, 0]  # (1, 1, H, W) -> (H, W)
        hover_pred = hover_pred[0].transpose(1, 2, 0)  # (1, 2, H, W) -> (H, W, 2)
        
        # Ensure correct shape
        if nuclear_pred.shape != original_shape:
            # Resize if needed
            nuclear_pred = cv2.resize(nuclear_pred, (original_shape[1], original_shape[0]))
            hover_pred = cv2.resize(hover_pred, (original_shape[1], original_shape[0]))
        
        # Binary nuclear mask
        nuclear_binary = (nuclear_pred > self.threshold).astype(np.uint8)
        
        result = {
            'nuclear': nuclear_pred,
            'hover': hover_pred,
            'nuclear_binary': nuclear_binary
        }
        
        # Generate instance map if requested
        if return_instances:
            instance_map = get_instance_map_from_predictions(
                nuclear_pred,
                hover_pred,
                threshold=self.threshold
            )
            result['instance'] = instance_map
        
        return result
    
    def predict_batch(
        self,
        images: list,
        return_instances: bool = True
    ) -> list:
        """
        Predict on a batch of images.
        
        Args:
            images: List of input images
            return_instances: Whether to return instance maps
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image in images:
            result = self.predict(image, return_instances=return_instances)
            results.append(result)
        return results
    
    def process_image_file(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        save_visualization: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Process a single image file and optionally save results.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save results (optional)
            save_visualization: Whether to save visualization
            
        Returns:
            Prediction dictionary
        """
        image_path = Path(image_path)
        
        # Predict
        predictions = self.predict(image_path, return_instances=True)
        
        # Save results if requested
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            stem = image_path.stem
            
            # Save nuclear segmentation
            nuclear_path = output_dir / f"{stem}_nuclear.png"
            cv2.imwrite(str(nuclear_path), (predictions['nuclear'] * 255).astype(np.uint8))
            
            # Save instance map
            instance_path = output_dir / f"{stem}_instances.png"
            # Use colormap for better visualization
            instance_colored = self._colorize_instance_map(predictions['instance'])
            cv2.imwrite(str(instance_path), instance_colored)
            
            # Save visualization if requested
            if save_visualization:
                vis_path = output_dir / f"{stem}_visualization.png"
                self._save_visualization(image_path, predictions, vis_path)
        
        return predictions
    
    def _colorize_instance_map(self, instance_map: np.ndarray) -> np.ndarray:
        """Colorize instance map for visualization."""
        # Create colored image
        colored = np.zeros((*instance_map.shape, 3), dtype=np.uint8)
        
        # Get unique instances
        unique_ids = np.unique(instance_map)
        unique_ids = unique_ids[unique_ids > 0]
        
        # Assign colors
        for idx, instance_id in enumerate(unique_ids):
            mask = (instance_map == instance_id)
            # Use HSV color space for distinct colors
            hue = int((idx * 180) / len(unique_ids))
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2RGB)[0][0]
            colored[mask] = color
        
        return colored
    
    def _save_visualization(
        self,
        image_path: Path,
        predictions: Dict[str, np.ndarray],
        output_path: Path
    ):
        """Save visualization of predictions."""
        from ..evaluation.visualizer import visualize_annotations
        
        # Load original image
        image = np.array(Image.open(image_path))
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Create visualization
        masks = {
            'nuclear': predictions['nuclear_binary'],
            'instance': predictions['instance']
        }
        
        visualize_annotations(
            image,
            masks,
            save_path=str(output_path),
            show=False
        )


if __name__ == '__main__':
    # Test inference
    print("Inference module loaded successfully")
