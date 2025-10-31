#!/usr/bin/env python3
"""
Complete pipeline script: data visualization -> training -> prediction visualization -> comparison
"""

import os
import sys
import subprocess
import argparse
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import paths


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed with return code {result.returncode}")
            return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run complete training pipeline")
    parser.add_argument("--test-epochs", type=int, default=2, 
                       help="Number of epochs for test training (default: 2)")
    parser.add_argument("--full-epochs", type=int, default=50, 
                       help="Number of epochs for full training (default: 50)")
    parser.add_argument("--skip-data-viz", action="store_true", 
                       help="Skip data visualization step")
    parser.add_argument("--skip-test-training", action="store_true", 
                       help="Skip test training step")
    parser.add_argument("--skip-full-training", action="store_true", 
                       help="Skip full training step")
    parser.add_argument("--skip-prediction-viz", action="store_true", 
                       help="Skip prediction visualization step")
    parser.add_argument("--models", nargs="+", default=None, 
                       help="Specific models to train (default: all)")
    
    args = parser.parse_args()
    
    print("🚀 Starting Complete Training Pipeline")
    print(f"Test epochs: {args.test_epochs}")
    print(f"Full epochs: {args.full_epochs}")
    if args.models:
        print(f"Models: {', '.join(args.models)}")
    else:
        print("Models: all available models")
    
    pipeline_start = time.time()
    results = {}
    
    # Step 1: Data Visualization
    if not args.skip_data_viz:
        cmd = [sys.executable, "scripts/visualize_data.py", "--num-samples", "3"]
        results["data_viz"] = run_command(cmd, "Data Visualization")
    else:
        print("⏭️  Skipping data visualization")
        results["data_viz"] = "skipped"
    
    # Step 2: Test Training (2 epochs)
    if not args.skip_test_training:
        cmd = [sys.executable, "scripts/train_all_models.py", "--epochs", str(args.test_epochs)]
        if args.models:
            cmd.extend(["--models"] + args.models)
        results["test_training"] = run_command(cmd, f"Test Training ({args.test_epochs} epochs)")
    else:
        print("⏭️  Skipping test training")
        results["test_training"] = "skipped"
    
    # Step 3: Full Training (50 epochs)
    if not args.skip_full_training:
        cmd = [sys.executable, "scripts/train_all_models.py", "--epochs", str(args.full_epochs)]
        if args.models:
            cmd.extend(["--models"] + args.models)
        results["full_training"] = run_command(cmd, f"Full Training ({args.full_epochs} epochs)")
    else:
        print("⏭️  Skipping full training")
        results["full_training"] = "skipped"
    
    # Step 4: Prediction Visualization
    if not args.skip_prediction_viz:
        cmd = [sys.executable, "scripts/visualize_predictions.py", "--num-samples", "2"]
        if args.models:
            cmd.extend(["--models"] + args.models)
        results["prediction_viz"] = run_command(cmd, "Prediction Visualization")
    else:
        print("⏭️  Skipping prediction visualization")
        results["prediction_viz"] = "skipped"
    
    # Pipeline Summary
    total_time = time.time() - pipeline_start
    print(f"\n{'='*60}")
    print("🎉 PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Total pipeline time: {total_time/60:.1f} minutes")
    print("\nStep Results:")
    
    for step, result in results.items():
        if result == "skipped":
            print(f"⏭️  {step}: skipped")
        elif result:
            print(f"✅ {step}: success")
        else:
            print(f"❌ {step}: failed")
    
    print(f"\n📁 Check these directories for results:")
    print(f"  - Data visualizations: {os.path.join(paths.logs, 'data_visualizations')}")
    print(f"  - Model checkpoints: {paths.models}")
    print(f"  - Training logs: {paths.logs}")
    print(f"  - Prediction visualizations: {os.path.join(paths.logs, 'prediction_visualizations')}")
    print(f"  - Training reports: {paths.logs}")
    
    print(f"\n📊 View training progress with TensorBoard:")
    print(f"  ./pika/Scripts/tensorboard.exe --logdir {os.path.join(paths.logs, 'tensorboard')}")
    
    # Check if all steps succeeded
    failed_steps = [step for step, result in results.items() if result is False]
    if failed_steps:
        print(f"\n⚠️  Some steps failed: {', '.join(failed_steps)}")
        return 1
    else:
        print(f"\n🎉 All steps completed successfully!")
        return 0


if __name__ == "__main__":
    exit(main())
