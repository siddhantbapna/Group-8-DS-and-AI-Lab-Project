#!/usr/bin/env python3
"""
Train all models sequentially with comprehensive logging and comparison
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd

# Add src to path
sys.path.append('src')

from config.config import Config, DataConfig, ModelConfig, TrainingConfig, SystemConfig
from src.train import Trainer, CrossValidationTrainer
from src.models import get_available_models, get_model_info

class AllModelsTrainer:
    """
    Train all available models sequentially with comprehensive tracking
    """
    
    def __init__(self, base_config: Config = None):
        self.base_config = base_config or Config()
        self.available_models = get_available_models()
        self.training_results = {}
        self.start_time = time.time()
        
        # Setup logging
        self.setup_logging()
        
        # Create results directory
        self.results_dir = "training_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.logger.info(f"Initialized AllModelsTrainer with {len(self.available_models)} models")
        self.logger.info(f"Available models: {self.available_models}")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_file = os.path.join(self.results_dir, f"all_models_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("AllModelsTrainer")
    
    def create_model_config(self, model_name: str) -> Config:
        """Create configuration for specific model"""
        config = Config()
        
        # Update model configuration
        config.model.model_name = model_name
        
        # Model-specific optimizations
        if model_name == "nnunet":
            # nnUNet is larger, use smaller batch size
            config.training.batch_size = 1
            config.training.num_epochs = 80  # Fewer epochs for larger model
        elif model_name == "attentionunet":
            # Attention UNet is also large
            config.training.batch_size = 1
            config.training.num_epochs = 90
        elif model_name == "vnet":
            # VNet is large
            config.training.batch_size = 1
            config.training.num_epochs = 90
        else:
            # Standard models
            config.training.batch_size = 2
            config.training.num_epochs = 100
        
        # Use optimized settings from EDA
        config.data.brain_only_training = True
        config.data.brain_mask_method = "otsu"
        config.data.background_weight = 0.05
        config.data.foreground_sampling = True
        config.training.loss_function = "weighted_dice_bce"
        config.training.scheduler = "poly"
        config.training.optimizer = "adamw"
        
        return config
    
    def train_single_model(self, model_name: str, use_cv: bool = True) -> Dict[str, Any]:
        """Train a single model"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Starting training for model: {model_name}")
        self.logger.info(f"{'='*60}")
        
        # Create model-specific configuration
        config = self.create_model_config(model_name)
        
        # Get model info
        try:
            model_info = get_model_info(model_name)
            self.logger.info(f"Model info: {model_info}")
        except Exception as e:
            self.logger.warning(f"Could not get model info: {e}")
            model_info = {"name": model_name, "total_parameters": "unknown"}
        
        # Record start time
        model_start_time = time.time()
        
        try:
            if use_cv:
                # Cross-validation training
                self.logger.info(f"Training {model_name} with {config.data.n_folds}-fold cross-validation")
                cv_trainer = CrossValidationTrainer(config)
                cv_trainer.train_all_folds()
                
                # Collect results from all folds
                fold_results = []
                for fold_result in cv_trainer.fold_results:
                    fold_results.append({
                        'fold': fold_result['fold'],
                        'best_metric': fold_result['best_metric'],
                        'final_epoch': len(fold_result['training_history'])
                    })
                
                # Calculate average performance
                dice_scores = [result['best_metric'] for result in fold_results]
                avg_dice = sum(dice_scores) / len(dice_scores)
                std_dice = (sum([(x - avg_dice) ** 2 for x in dice_scores]) / len(dice_scores)) ** 0.5
                
                result = {
                    'model_name': model_name,
                    'model_info': model_info,
                    'training_type': 'cross_validation',
                    'n_folds': config.data.n_folds,
                    'fold_results': fold_results,
                    'average_dice': avg_dice,
                    'std_dice': std_dice,
                    'best_fold': max(fold_results, key=lambda x: x['best_metric']),
                    'worst_fold': min(fold_results, key=lambda x: x['best_metric']),
                    'training_time': time.time() - model_start_time,
                    'config': {
                        'batch_size': config.training.batch_size,
                        'num_epochs': config.training.num_epochs,
                        'learning_rate': config.training.learning_rate,
                        'loss_function': config.training.loss_function,
                        'optimizer': config.training.optimizer,
                        'scheduler': config.training.scheduler
                    }
                }
            else:
                # Single fold training
                self.logger.info(f"Training {model_name} on single fold")
                trainer = Trainer(config, fold=0)
                trainer.train()
                
                result = {
                    'model_name': model_name,
                    'model_info': model_info,
                    'training_type': 'single_fold',
                    'best_metric': trainer.best_metric,
                    'training_time': time.time() - model_start_time,
                    'config': {
                        'batch_size': config.training.batch_size,
                        'num_epochs': config.training.num_epochs,
                        'learning_rate': config.training.learning_rate,
                        'loss_function': config.training.loss_function,
                        'optimizer': config.training.optimizer,
                        'scheduler': config.training.scheduler
                    }
                }
            
            self.logger.info(f"✅ {model_name} training completed successfully!")
            self.logger.info(f"Training time: {result['training_time']:.2f} seconds")
            
            if use_cv:
                self.logger.info(f"Average Dice Score: {result['average_dice']:.4f} ± {result['std_dice']:.4f}")
                self.logger.info(f"Best fold: {result['best_fold']['fold']} (Dice: {result['best_fold']['best_metric']:.4f})")
            else:
                self.logger.info(f"Best Dice Score: {result['best_metric']:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error training {model_name}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            return {
                'model_name': model_name,
                'model_info': model_info,
                'training_type': 'cross_validation' if use_cv else 'single_fold',
                'error': str(e),
                'training_time': time.time() - model_start_time,
                'success': False
            }
    
    def train_all_models(self, use_cv: bool = True, models_to_train: List[str] = None):
        """Train all models sequentially"""
        if models_to_train is None:
            models_to_train = self.available_models
        
        self.logger.info(f"\n🚀 Starting training of {len(models_to_train)} models")
        self.logger.info(f"Models to train: {models_to_train}")
        self.logger.info(f"Cross-validation: {'Yes' if use_cv else 'No'}")
        self.logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        for i, model_name in enumerate(models_to_train, 1):
            self.logger.info(f"\n📊 Progress: {i}/{len(models_to_train)} models")
            
            # Train the model
            result = self.train_single_model(model_name, use_cv)
            self.training_results[model_name] = result
            
            # Save intermediate results
            self.save_results()
            
            # Log progress
            elapsed_time = time.time() - self.start_time
            remaining_models = len(models_to_train) - i
            if i > 0:
                avg_time_per_model = elapsed_time / i
                estimated_remaining = avg_time_per_model * remaining_models
                self.logger.info(f"⏱️  Elapsed: {elapsed_time/3600:.2f}h, Estimated remaining: {estimated_remaining/3600:.2f}h")
        
        # Final summary
        self.generate_final_summary()
    
    def save_results(self):
        """Save training results to JSON"""
        results_file = os.path.join(self.results_dir, "training_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.training_results, f, indent=2, default=str)
        
        # Also save as CSV for easy analysis
        self.save_results_csv()
    
    def save_results_csv(self):
        """Save results as CSV for analysis"""
        if not self.training_results:
            return
        
        rows = []
        for model_name, result in self.training_results.items():
            if result.get('success', True):  # Skip failed models
                row = {
                    'model_name': model_name,
                    'total_parameters': result.get('model_info', {}).get('total_parameters', 'unknown'),
                    'training_type': result.get('training_type', 'unknown'),
                    'training_time_hours': result.get('training_time', 0) / 3600,
                }
                
                if result.get('training_type') == 'cross_validation':
                    row.update({
                        'average_dice': result.get('average_dice', 0),
                        'std_dice': result.get('std_dice', 0),
                        'best_fold_dice': result.get('best_fold', {}).get('best_metric', 0),
                        'worst_fold_dice': result.get('worst_fold', {}).get('best_metric', 0),
                        'n_folds': result.get('n_folds', 0)
                    })
                else:
                    row.update({
                        'best_dice': result.get('best_metric', 0)
                    })
                
                rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            csv_file = os.path.join(self.results_dir, "training_results.csv")
            df.to_csv(csv_file, index=False)
            self.logger.info(f"Results saved to CSV: {csv_file}")
    
    def generate_final_summary(self):
        """Generate final training summary"""
        total_time = time.time() - self.start_time
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🎉 ALL MODELS TRAINING COMPLETED!")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Total training time: {total_time/3600:.2f} hours")
        self.logger.info(f"Models trained: {len(self.training_results)}")
        
        # Summary table
        self.logger.info(f"\n📊 FINAL RESULTS SUMMARY:")
        self.logger.info(f"{'Model':<15} {'Parameters':<12} {'Avg Dice':<10} {'Std':<8} {'Time(h)':<8}")
        self.logger.info(f"{'-'*60}")
        
        for model_name, result in self.training_results.items():
            if result.get('success', True):
                params = result.get('model_info', {}).get('total_parameters', 'unknown')
                if isinstance(params, int):
                    params = f"{params:,}"
                
                if result.get('training_type') == 'cross_validation':
                    avg_dice = result.get('average_dice', 0)
                    std_dice = result.get('std_dice', 0)
                    self.logger.info(f"{model_name:<15} {params:<12} {avg_dice:<10.4f} {std_dice:<8.4f} {result.get('training_time', 0)/3600:<8.2f}")
                else:
                    best_dice = result.get('best_metric', 0)
                    self.logger.info(f"{model_name:<15} {params:<12} {best_dice:<10.4f} {'N/A':<8} {result.get('training_time', 0)/3600:<8.2f}")
        
        # Best model
        if self.training_results:
            best_model = max(
                [(name, result) for name, result in self.training_results.items() if result.get('success', True)],
                key=lambda x: x[1].get('average_dice', x[1].get('best_metric', 0))
            )
            self.logger.info(f"\n🏆 BEST MODEL: {best_model[0]} (Dice: {best_model[1].get('average_dice', best_model[1].get('best_metric', 0)):.4f})")
        
        self.logger.info(f"\n📁 Results saved to: {self.results_dir}/")
        self.logger.info(f"   - training_results.json (detailed results)")
        self.logger.info(f"   - training_results.csv (summary table)")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train all models sequentially')
    parser.add_argument('--models', nargs='+', default=None,
                       help='Specific models to train (default: all)')
    parser.add_argument('--no-cv', action='store_true',
                       help='Disable cross-validation (single fold only)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with fewer epochs')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = AllModelsTrainer()
    
    # Quick test mode
    if args.quick:
        trainer.logger.info("🚀 Quick test mode - using fewer epochs")
        # Override config for quick testing
        original_config = trainer.base_config
        original_config.training.num_epochs = 5
        original_config.data.n_folds = 2
    
    # Train models
    use_cv = not args.no_cv
    models_to_train = args.models if args.models else None
    
    trainer.train_all_models(use_cv=use_cv, models_to_train=models_to_train)

if __name__ == "__main__":
    main()
