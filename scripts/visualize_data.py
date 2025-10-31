#!/usr/bin/env python3
"""
Data visualization script for pre-training inspection
"""

import os
import sys
import glob
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import paths
from src.visualization import visualize_data_sample


def find_sample_cases(data_dir: str, num_samples: int = 3) -> list:
    """Find sample cases for visualization"""
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return []
    
    case_dirs = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            # Check if it has required files with flexible patterns
            required_patterns = [
                ["*t1n.nii.gz", "*t1n.nii", "*t1*.nii.gz", "*t1*.nii"],
                ["*t1c.nii.gz", "*t1c.nii", "*t1ce*.nii.gz", "*t1ce*.nii"],
                ["*t2w.nii.gz", "*t2w.nii", "*t2*.nii.gz", "*t2*.nii"],
                ["*t2f.nii.gz", "*t2f.nii", "*flair*.nii.gz", "*flair*.nii"],
                ["*seg.nii.gz", "*seg.nii", "*seg*.nii.gz", "*seg*.nii"]
            ]
            has_all = True
            for patterns in required_patterns:
                found = False
                for pattern in patterns:
                    if glob.glob(os.path.join(item_path, pattern)):
                        found = True
                        break
                if not found:
                    has_all = False
                    break
            
            if has_all:
                case_dirs.append(item_path)
    
    # Return first num_samples cases
    return case_dirs[:num_samples]


def main():
    parser = argparse.ArgumentParser(description="Visualize data samples before training")
    parser.add_argument("--data-dir", type=str, default=paths.data_train, 
                       help="Path to training data directory")
    parser.add_argument("--num-samples", type=int, default=3, 
                       help="Number of sample cases to visualize")
    parser.add_argument("--save-dir", type=str, default=os.path.join(paths.logs, "data_visualizations"),
                       help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Find sample cases
    print(f"Looking for sample cases in: {args.data_dir}")
    sample_cases = find_sample_cases(args.data_dir, args.num_samples)
    
    if not sample_cases:
        print("No valid sample cases found!")
        return
    
    print(f"Found {len(sample_cases)} sample cases")
    
    # Visualize each sample
    for i, case_dir in enumerate(sample_cases):
        case_name = os.path.basename(case_dir)
        save_path = os.path.join(args.save_dir, f"data_sample_{i+1}_{case_name}.png")
        
        print(f"\nVisualizing case {i+1}/{len(sample_cases)}: {case_name}")
        visualize_data_sample(case_dir, save_path)
    
    print(f"\nData visualization complete!")
    print(f"Visualizations saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
