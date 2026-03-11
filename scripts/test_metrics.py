"""
Test script for evaluation metrics
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.evaluation.metrics import evaluate_predictions

def test_metrics():
    """Test evaluation metrics."""
    print("Testing evaluation metrics...")
    
    # Create dummy data
    h, w = 256, 256
    nuclear_pred = np.random.rand(h, w)
    hover_pred = np.random.randn(h, w, 2)
    nuclear_target = np.random.randint(0, 2, (h, w))
    instance_target = np.random.randint(0, 5, (h, w))
    
    # Evaluate
    results = evaluate_predictions(
        nuclear_pred, hover_pred, nuclear_target, instance_target
    )
    
    print("\nEvaluation Results:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\nMetrics loaded successfully!")

if __name__ == '__main__':
    test_metrics()
