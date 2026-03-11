"""
Test TensorBoard Logging
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from torch.utils.tensorboard import SummaryWriter
    import torch
    import numpy as np
    
    print("TensorBoard is available!")
    
    # Test writing logs
    log_dir = project_root / "outputs" / "logs" / "test"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # Write some test data
    for i in range(10):
        writer.add_scalar('Test/Loss', np.random.rand(), i)
        writer.add_scalar('Test/Accuracy', np.random.rand(), i)
    
    writer.close()
    
    print(f"Test logs written to: {log_dir}")
    print("\nTo view in TensorBoard, run:")
    print(f"  tensorboard --logdir {log_dir.parent}")
    print("\nThen open: http://localhost:6006")
    
except ImportError:
    print("ERROR: TensorBoard is not installed!")
    print("Install with: pip install tensorboard")
