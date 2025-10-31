#!/usr/bin/env python3
"""
Post-training prediction visualization script
"""

import os
import sys
import glob
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import paths
from src.visualization import visualize_predictions


def find_trained_models() -> list:
    """Find all trained models"""
    models = []
    model_pattern = os.path.join(paths.models, "best_*.pth")
    model_files = glob.glob(model_pattern)
    
    for model_file in model_files:
        model_name = os.path.basename(model_file).replace("best_", "").replace(".pth", "")
        models.append((model_name, model_file))
    
    return models


def find_sample_cases(data_dir: str, num_samples: int = 2) -> list:
    """Find sample cases for prediction visualization"""
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return []
    
    case_dirs = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            # Check if it has required files
            required_files = ["*t1.nii.gz", "*t1ce.nii.gz", "*t2.nii.gz", "*flair.nii.gz", "*seg.nii.gz"]
            has_all = True
            for pattern in required_files:
                if not glob.glob(os.path.join(item_path, pattern)):
                    has_all = False
                    break
            
            if has_all:
                case_dirs.append(item_path)
    
    # Return first num_samples cases
    return case_dirs[:num_samples]


def main():
    parser = argparse.ArgumentParser(description="Visualize model predictions on validation data")
    parser.add_argument("--data-dir", type=str, default=paths.data_train, 
                       help="Path to training data directory")
    parser.add_argument("--num-samples", type=int, default=2, 
                       help="Number of sample cases to visualize")
    parser.add_argument("--save-dir", type=str, default=os.path.join(paths.logs, "prediction_visualizations"),
                       help="Directory to save visualizations")
    parser.add_argument("--models", nargs="+", default=None, 
                       help="Specific models to visualize (default: all trained models)")
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Find trained models
    trained_models = find_trained_models()
    
    if not trained_models:
        print("No trained models found!")
        return
    
    # Filter models if specified
    if args.models:
        trained_models = [(name, path) for name, path in trained_models if name in args.models]
    
    print(f"Found {len(trained_models)} trained models: {[name for name, _ in trained_models]}")
    
    # Find sample cases
    print(f"Looking for sample cases in: {args.data_dir}")
    sample_cases = find_sample_cases(args.data_dir, args.num_samples)
    
    if not sample_cases:
        print("No valid sample cases found!")
        return
    
    print(f"Found {len(sample_cases)} sample cases")
    
    # Visualize predictions for each model-case combination
    for model_name, model_path in trained_models:
        print(f"\n{'='*60}")
        print(f"Visualizing predictions for model: {model_name}")
        print(f"{'='*60}")
        
        for i, case_dir in enumerate(sample_cases):
            case_name = os.path.basename(case_dir)
            save_path = os.path.join(args.save_dir, f"predictions_{model_name}_{case_name}.png")
            
            print(f"\nProcessing case {i+1}/{len(sample_cases)}: {case_name}")
            
            try:
                visualize_predictions(model_name, case_dir, model_path, save_path)
                print(f"Predictions saved to: {save_path}")
            except Exception as e:
                print(f"Failed to visualize predictions for {model_name} on {case_name}: {e}")
    
    print(f"\nPrediction visualization complete!")
    print(f"Visualizations saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
