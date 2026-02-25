"""
Evaluation Script for HoVer-Net
Evaluates model on test set and computes metrics
"""

import sys
import argparse
import yaml
from pathlib import Path
from typing import Optional
import torch
import numpy as np
from tqdm import tqdm
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import HoVerNet
from src.data.dataloader import get_dataloaders
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.visualizer import visualize_predictions
import matplotlib.pyplot as plt


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_model(
    model: HoVerNet,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    save_predictions: bool = False,
    output_dir: Optional[Path] = None,
    num_samples: Optional[int] = None
) -> dict:
    """
    Evaluate model on dataset.
    
    Args:
        model: Trained HoVer-Net model
        dataloader: DataLoader for evaluation
        device: Device to use
        save_predictions: Whether to save prediction visualizations
        output_dir: Directory to save predictions
        num_samples: Number of samples to evaluate (None = all)
        
    Returns:
        Dictionary with average metrics
    """
    model.eval()
    
    all_metrics = []
    
    if save_predictions and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Evaluating')):
            if num_samples and batch_idx >= num_samples:
                break
            
            # Move to device
            images = batch['image'].to(device)
            nuclear_target = batch['nuclear'].cpu().numpy()
            instance_target = batch['instance'].cpu().numpy()
            
            # Forward pass
            predictions = model(images)
            
            # Process each sample in batch
            for i in range(images.shape[0]):
                nuclear_pred = torch.sigmoid(predictions['nuclear'][i]).cpu().numpy()
                hover_pred = predictions['hover'][i].cpu().numpy()
                
                # Convert from (C, H, W) to (H, W) or (H, W, C)
                if nuclear_pred.ndim == 3:
                    nuclear_pred = nuclear_pred[0]  # (1, H, W) -> (H, W)
                
                if hover_pred.ndim == 3:
                    hover_pred = hover_pred.transpose(1, 2, 0)  # (2, H, W) -> (H, W, 2)
                
                # Get targets for this sample
                nuc_target = nuclear_target[i]
                inst_target = instance_target[i]
                
                # Evaluate
                metrics = evaluate_predictions(
                    nuclear_pred,
                    hover_pred,
                    nuc_target,
                    inst_target
                )
                
                all_metrics.append(metrics)
                
                # Save visualization if requested
                if save_predictions and output_dir:
                    image = images[i].cpu().numpy().transpose(1, 2, 0)
                    
                    # Get instance map from predictions
                    from src.evaluation.metrics import get_instance_map_from_predictions
                    pred_instances = get_instance_map_from_predictions(
                        nuclear_pred, hover_pred
                    )
                    
                    # Visualize
                    pred_dict = {
                        'nuclear': (nuclear_pred > 0.5).astype(np.uint8),
                        'instance': pred_instances
                    }
                    target_dict = {
                        'nuclear': nuc_target.astype(np.uint8),
                        'instance': inst_target
                    }
                    
                    sample_name = batch['name'][i] if 'name' in batch else f'sample_{batch_idx}_{i}'
                    save_path = output_dir / f'{sample_name}_prediction.png'
                    
                    visualize_predictions(
                        image,
                        target_dict,
                        pred_dict,
                        save_path=str(save_path),
                        show=False
                    )
    
    # Compute average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        if isinstance(all_metrics[0][key], (int, float)):
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        else:
            avg_metrics[key] = all_metrics[0][key]
    
    return avg_metrics, all_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate HoVer-Net')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Which split to evaluate on'
    )
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save prediction visualizations'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Number of samples to evaluate (None = all)'
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
    
    # Data loader
    print(f"\nLoading {args.split} data...")
    dataloaders = get_dataloaders(
        data_dir=config['data']['data_dir'],
        batch_size=1,  # Evaluate one at a time for visualization
        num_workers=0,
        split_file=config['data']['split_file'],
        cache_masks=False,
        generate_hover=config['data']['generate_hover']
    )
    
    dataloader = dataloaders[args.split]
    print(f"Evaluating on {len(dataloader.dataset)} samples")
    
    # Model
    print("\nLoading model...")
    model = HoVerNet(
        backbone=config['model']['backbone'],
        pretrained=False,  # Not needed for inference
        num_types=config['model']['num_types'],
        decoder_channels=config['model']['decoder_channels']
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Output directory
    output_dir = None
    if args.save_predictions:
        output_dir = Path('outputs') / 'predictions' / args.split
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving predictions to {output_dir}")
    
    # Evaluate
    print("\n" + "="*60)
    print("Evaluating model...")
    print("="*60 + "\n")
    
    avg_metrics, all_metrics = evaluate_model(
        model,
        dataloader,
        device=device,
        save_predictions=args.save_predictions,
        output_dir=output_dir,
        num_samples=args.num_samples
    )
    
    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"\nAverage Metrics:")
    print(f"  Dice Score:        {avg_metrics['dice']:.4f}")
    print(f"  Pixel Accuracy:   {avg_metrics['pixel_accuracy']:.4f}")
    print(f"  AJI:              {avg_metrics['aji']:.4f}")
    print(f"  Precision:        {avg_metrics['precision']:.4f}")
    print(f"  Recall:           {avg_metrics['recall']:.4f}")
    print(f"  F1 Score:         {avg_metrics['f1']:.4f}")
    print(f"  Panoptic Quality: {avg_metrics['pq']:.4f}")
    print(f"  Segmentation Q:   {avg_metrics['sq']:.4f}")
    print(f"  Detection Q:      {avg_metrics['dq']:.4f}")
    print(f"\n  Predicted Instances: {avg_metrics['num_pred_instances']:.1f}")
    print(f"  Target Instances:    {avg_metrics['num_target_instances']:.1f}")
    
    # Save results
    results_path = Path('outputs') / 'evaluation_results.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'checkpoint': args.checkpoint,
        'split': args.split,
        'num_samples': len(all_metrics),
        'average_metrics': avg_metrics,
        'per_sample_metrics': all_metrics
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print("="*60)


if __name__ == '__main__':
    main()
