#!/usr/bin/env python3
"""
Automated training script for all models
Runs all models for specified epochs and generates comparison reports
"""

import os
import sys
import subprocess
import time
from typing import List, Dict
import json
import glob

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import paths
from src.visualization import compare_models_performance


def run_training(model_name: str, epochs: int, spatial_dims: int = 3) -> bool:
    """Run training for a single model"""
    print(f"\n{'='*60}")
    print(f"Training {model_name} for {epochs} epochs")
    print(f"{'='*60}")
    
    # Update config for specified epochs
    from config.config import train_cfg
    original_epochs = train_cfg.max_epochs
    train_cfg.max_epochs = epochs
    
    try:
        # Run training
        cmd = [
            sys.executable, "main.py",
            "--model", model_name,
            "--mode", "train",
            "--epochs", str(epochs)
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        # Stream live stdout so user sees progress bars and device logs
        with subprocess.Popen(
            cmd,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as proc:
            for line in proc.stdout:
                print(line, end="")
            proc.wait()
            if proc.returncode == 0:
                print(f"[SUCCESS] {model_name} training completed successfully")
                return True
            else:
                print(f"[FAILED] {model_name} training failed")
                return False
            
    except Exception as e:
        print(f"[FAILED] {model_name} training failed with exception: {e}")
        return False
    finally:
        # Restore original epochs
        train_cfg.max_epochs = original_epochs


def get_available_models() -> List[str]:
    """Get list of available models"""
    # return ["unet2d", "unet3d", "attenunet", "nnunet", "resunet", "vnet"]
    return ["unet3d", "resunet", "attenunet", "nnunet","vnet"]


def check_model_completion(model_name: str) -> bool:
    """Check if model training is complete"""
    best_path = os.path.join(paths.models, f"best_{model_name}.pth")
    history_path = os.path.join(paths.logs, f"training_history_{model_name}.json")
    
    return os.path.exists(best_path) and os.path.exists(history_path)


def generate_training_report(epochs: int) -> str:
    """Generate comprehensive training report"""
    models = get_available_models()
    report_path = os.path.join(paths.logs, f"training_report_{epochs}epochs.md")
    
    with open(report_path, 'w') as f:
        f.write(f"# Training Report - {epochs} Epochs\n\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Model Status\n\n")
        f.write("| Model | Status | Best Model | History |\n")
        f.write("|-------|--------|------------|----------|\n")
        
        for model in models:
            status = "[SUCCESS] Complete" if check_model_completion(model) else "[FAILED] Failed/Incomplete"
            best_exists = "[SUCCESS]" if os.path.exists(os.path.join(paths.models, f"best_{model}.pth")) else "[FAILED]"
            history_exists = "[SUCCESS]" if os.path.exists(os.path.join(paths.logs, f"training_history_{model}.json")) else "[FAILED]"
            
            f.write(f"| {model} | {status} | {best_exists} | {history_exists} |\n")
        
        f.write("\n## Performance Summary\n\n")
        
        # Load performance data
        performance_data = []
        for model in models:
            history_pattern = os.path.join(paths.logs, f"training_history_{model}.json")
            history_files = glob.glob(history_pattern)
            
            if history_files:
                with open(history_files[0], 'r') as hf:
                    history = json.load(hf)
                
                best_dice = max(history['dice']) if history['dice'] else 0
                final_dice = history['dice'][-1] if history['dice'] else 0
                best_epoch = history['dice'].index(best_dice) + 1 if history['dice'] else 0
                final_train_loss = history['train_loss'][-1] if history['train_loss'] else 0
                final_val_loss = history['val_loss'][-1] if history['val_loss'] else 0
                
                performance_data.append({
                    'model': model,
                    'best_dice': best_dice,
                    'final_dice': final_dice,
                    'best_epoch': best_epoch,
                    'final_train_loss': final_train_loss,
                    'final_val_loss': final_val_loss
                })
        
        if performance_data:
            # Sort by best dice score
            performance_data.sort(key=lambda x: x['best_dice'], reverse=True)
            
            f.write("| Rank | Model | Best Dice | Final Dice | Best Epoch | Final Train Loss | Final Val Loss |\n")
            f.write("|------|-------|-----------|------------|------------|------------------|----------------|\n")
            
            for i, data in enumerate(performance_data, 1):
                f.write(f"| {i} | {data['model']} | {data['best_dice']:.4f} | {data['final_dice']:.4f} | "
                       f"{data['best_epoch']} | {data['final_train_loss']:.4f} | {data['final_val_loss']:.4f} |\n")
            
            f.write(f"\n## Best Performing Model\n\n")
            f.write(f"**{performance_data[0]['model']}** achieved the highest Dice score of **{performance_data[0]['best_dice']:.4f}**\n\n")
        
        f.write("## Files Generated\n\n")
        f.write("- Model checkpoints: `outputs/models/best_*.pth`\n")
        f.write("- Training histories: `outputs/logs/training_history_*.json`\n")
        f.write("- TensorBoard logs: `outputs/logs/tensorboard/`\n")
        f.write("- Training logs: `outputs/logs/*.log`\n")
        f.write("- Comparison plots: `outputs/logs/model_comparison_*.png`\n")
    
    return report_path


def main():
    """Main training orchestration"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train all models for comparison")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train (default: 2)")
    parser.add_argument("--models", nargs="+", default=None, help="Specific models to train (default: all)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip models that already have checkpoints")
    parser.add_argument("--generate-report", action="store_true", default=True, help="Generate training report")
    
    args = parser.parse_args()
    
    # Get models to train
    if args.models:
        models_to_train = args.models
    else:
        models_to_train = get_available_models()
    
    print(f"Training {len(models_to_train)} models for {args.epochs} epochs")
    print(f"Models: {', '.join(models_to_train)}")
    
    # Track results
    results = {}
    start_time = time.time()
    
    for model in models_to_train:
        # Check if model already exists
        if args.skip_existing and check_model_completion(model):
            print(f"[SKIP] Skipping {model} - already completed")
            results[model] = "skipped"
            continue
        
        # Determine spatial dimensions
        spatial_dims = 2 if model == "unet2d" else 3
        
        # Run training
        success = run_training(model, args.epochs, spatial_dims)
        results[model] = "success" if success else "failed"
        
        # Small delay between models
        time.sleep(2)
    
    # Print summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Models trained: {len(models_to_train)}")
    
    for model, status in results.items():
        status_icon = "[SUCCESS]" if status == "success" else "[SKIPPED]" if status == "skipped" else "[FAILED]"
        print(f"{status_icon} {model}: {status}")
    
    # Generate comparison plots
    successful_models = [model for model, status in results.items() if status == "success"]
    if len(successful_models) > 1:
        print(f"\nGenerating comparison plots for {len(successful_models)} models...")
        comparison_path = os.path.join(paths.logs, f"model_comparison_{args.epochs}epochs.png")
        compare_models_performance(successful_models, comparison_path)
    
    # Generate report
    if args.generate_report:
        print("\nGenerating training report...")
        report_path = generate_training_report(args.epochs)
        print(f"Training report saved to: {report_path}")
    
    print(f"\nTraining complete! Check outputs/logs/ for results.")


if __name__ == "__main__":
    main()
