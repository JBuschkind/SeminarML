"""
Inference Script for HoVer-Net
Process new images and generate predictions
"""

import sys
import argparse
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.inference import HoVerNetInference
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='HoVer-Net Inference')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input image or directory containing images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/inference',
        help='Output directory for results'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Threshold for nuclear segmentation'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    parser.add_argument(
        '--no-visualization',
        action='store_true',
        help='Skip saving visualization'
    )
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Initialize inference
    print(f"Loading model from {args.checkpoint}...")
    inference = HoVerNetInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        threshold=args.threshold
    )
    
    # Process input
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_file():
        # Single image
        print(f"\nProcessing image: {input_path}")
        predictions = inference.process_image_file(
            input_path,
            output_dir=output_dir,
            save_visualization=not args.no_visualization
        )
        print(f"Results saved to {output_dir}")
        
    elif input_path.is_dir():
        # Directory of images
        image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
        image_files = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        print(f"\nProcessing {len(image_files)} images...")
        
        for image_file in tqdm(image_files, desc='Processing'):
            try:
                inference.process_image_file(
                    image_file,
                    output_dir=output_dir,
                    save_visualization=not args.no_visualization
                )
            except Exception as e:
                print(f"\nError processing {image_file}: {e}")
                continue
        
        print(f"\nAll results saved to {output_dir}")
        
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return
    
    print("\nInference completed!")


if __name__ == '__main__':
    main()
