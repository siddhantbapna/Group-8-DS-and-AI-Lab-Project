#!/usr/bin/env python3
"""
Quick training script for all models - simplified version
"""

import os
import sys
import subprocess
import time
from datetime import datetime

# Add src to path
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from models import get_available_models

def run_training_command(model_name: str, epochs: int = 10, use_cv: bool = False):
    """Run training command for a specific model"""
    cmd = [
        "pika\\Scripts\\python.exe", "main.py",
        "--model", model_name,
        "--epochs", str(epochs),
        "--batch_size", "2",
        "--learning_rate", "1e-4",
        "--loss_function", "weighted_dice_bce"
    ]
    
    if use_cv:
        cmd.extend(["--mode", "cv", "--n_folds", "3"])  # Use 3 folds for quick testing
    else:
        cmd.extend(["--mode", "train", "--fold", "0"])
    
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True)

def main():
    """Train all models quickly"""
    models = get_available_models()
    
    print(f"Quick Training All Models")
    print(f"Models: {models}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    results = {}
    start_time = time.time()
    
    for i, model in enumerate(models, 1):
        print(f"\nTraining {i}/{len(models)}: {model}")
        print("-" * 40)
        
        # Run training
        result = run_training_command(model, epochs=5, use_cv=False)  # Quick test
        
        if result.returncode == 0:
            print(f"SUCCESS: {model} completed successfully")
            results[model] = "SUCCESS"
        else:
            print(f"FAILED: {model} failed")
            print(f"Error: {result.stderr}")
            results[model] = "FAILED"
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"ALL MODELS TRAINING COMPLETED!")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Results: {results}")
    
    successful = sum(1 for status in results.values() if status == "SUCCESS")
    print(f"Successful: {successful}/{len(models)}")

if __name__ == "__main__":
    main()
