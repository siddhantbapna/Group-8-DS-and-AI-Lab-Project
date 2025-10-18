"""
Checkpoint management system for model training
"""
import os
import torch
import json
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import logging

class CheckpointManager:
    """
    Comprehensive checkpoint management system
    """
    
    def __init__(self, checkpoint_dir: str, model_name: str, 
                 save_best: bool = True, save_last: bool = True,
                 max_checkpoints: int = 5):
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.save_best = save_best
        self.save_last = save_last
        self.max_checkpoints = max_checkpoints
        
        # Create checkpoint directory
        self.model_checkpoint_dir = os.path.join(checkpoint_dir, model_name)
        os.makedirs(self.model_checkpoint_dir, exist_ok=True)
        
        # Initialize tracking variables
        self.best_metric = float('-inf')
        self.best_epoch = 0
        self.last_epoch = 0
        self.checkpoint_history = []
        
        # Setup logging
        self.logger = logging.getLogger(f"CheckpointManager_{model_name}")
        
    def save_checkpoint(self, epoch: int, model: torch.nn.Module, 
                       optimizer: torch.optim.Optimizer, 
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                       metrics: Dict[str, float], 
                       loss: float,
                       is_best: bool = False,
                       additional_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler state
            metrics: Validation metrics
            loss: Validation loss
            is_best: Whether this is the best model so far
            additional_info: Additional information to save
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint_info = {
            'epoch': epoch,
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'loss': loss,
            'is_best': is_best,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'additional_info': additional_info or {}
        }
        
        # Create checkpoint filename
        if is_best:
            checkpoint_filename = f"best_model_epoch_{epoch}.pth"
        else:
            checkpoint_filename = f"checkpoint_epoch_{epoch}.pth"
        
        checkpoint_path = os.path.join(self.model_checkpoint_dir, checkpoint_filename)
        
        # Save checkpoint
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'loss': loss,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'checkpoint_info': checkpoint_info
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        
        # Save checkpoint info as JSON
        info_path = checkpoint_path.replace('.pth', '_info.json')
        with open(info_path, 'w') as f:
            json.dump(checkpoint_info, f, indent=2)
        
        # Update tracking
        self.last_epoch = epoch
        self.checkpoint_history.append({
            'epoch': epoch,
            'path': checkpoint_path,
            'metrics': metrics,
            'loss': loss,
            'is_best': is_best,
            'timestamp': checkpoint_info['timestamp']
        })
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str, model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       device: str = 'cpu') -> Dict[str, Any]:
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            device: Device to load checkpoint on
        
        Returns:
            Dictionary with checkpoint information
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint_data['model_state_dict'])
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint_data:
            if checkpoint_data['scheduler_state_dict'] is not None:
                scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
        
        # Update tracking variables
        self.best_metric = checkpoint_data.get('best_metric', float('-inf'))
        self.best_epoch = checkpoint_data.get('best_epoch', 0)
        self.last_epoch = checkpoint_data.get('epoch', 0)
        
        self.logger.info(f"Loaded checkpoint: {checkpoint_path}")
        return checkpoint_data
    
    def load_best_checkpoint(self, model: torch.nn.Module,
                            optimizer: Optional[torch.optim.Optimizer] = None,
                            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                            device: str = 'cpu') -> Dict[str, Any]:
        """
        Load the best checkpoint
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            device: Device to load checkpoint on
        
        Returns:
            Dictionary with checkpoint information
        """
        best_checkpoint_path = self.get_best_checkpoint_path()
        if best_checkpoint_path is None:
            raise FileNotFoundError("No best checkpoint found")
        
        return self.load_checkpoint(best_checkpoint_path, model, optimizer, scheduler, device)
    
    def load_latest_checkpoint(self, model: torch.nn.Module,
                              optimizer: Optional[torch.optim.Optimizer] = None,
                              scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                              device: str = 'cpu') -> Dict[str, Any]:
        """
        Load the latest checkpoint
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            device: Device to load checkpoint on
        
        Returns:
            Dictionary with checkpoint information
        """
        latest_checkpoint_path = self.get_latest_checkpoint_path()
        if latest_checkpoint_path is None:
            raise FileNotFoundError("No checkpoint found")
        
        return self.load_checkpoint(latest_checkpoint_path, model, optimizer, scheduler, device)
    
    def get_best_checkpoint_path(self) -> Optional[str]:
        """Get path to best checkpoint"""
        best_checkpoints = [cp for cp in self.checkpoint_history if cp['is_best']]
        if not best_checkpoints:
            return None
        
        # Return the most recent best checkpoint
        best_checkpoints.sort(key=lambda x: x['epoch'], reverse=True)
        return best_checkpoints[0]['path']
    
    def get_latest_checkpoint_path(self) -> Optional[str]:
        """Get path to latest checkpoint"""
        if not self.checkpoint_history:
            return None
        
        # Return the most recent checkpoint
        latest_checkpoint = max(self.checkpoint_history, key=lambda x: x['epoch'])
        return latest_checkpoint['path']
    
    def get_checkpoint_info(self, checkpoint_path: str) -> Dict[str, Any]:
        """Get information about a checkpoint"""
        info_path = checkpoint_path.replace('.pth', '_info.json')
        if not os.path.exists(info_path):
            return {}
        
        with open(info_path, 'r') as f:
            return json.load(f)
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints"""
        return self.checkpoint_history.copy()
    
    def update_best_metric(self, metric_value: float, epoch: int):
        """Update best metric value"""
        if metric_value > self.best_metric:
            self.best_metric = metric_value
            self.best_epoch = epoch
            return True
        return False
    
    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints to save disk space"""
        if len(self.checkpoint_history) <= self.max_checkpoints:
            return
        
        # Sort by epoch (oldest first)
        sorted_checkpoints = sorted(self.checkpoint_history, key=lambda x: x['epoch'])
        
        # Remove old checkpoints (keep the most recent ones)
        checkpoints_to_remove = sorted_checkpoints[:-self.max_checkpoints]
        
        for checkpoint in checkpoints_to_remove:
            try:
                # Remove checkpoint file
                if os.path.exists(checkpoint['path']):
                    os.remove(checkpoint['path'])
                
                # Remove info file
                info_path = checkpoint['path'].replace('.pth', '_info.json')
                if os.path.exists(info_path):
                    os.remove(info_path)
                
                # Remove from history
                self.checkpoint_history.remove(checkpoint)
                
                self.logger.info(f"Removed old checkpoint: {checkpoint['path']}")
            except Exception as e:
                self.logger.warning(f"Failed to remove checkpoint {checkpoint['path']}: {e}")
    
    def save_model_summary(self, model: torch.nn.Module, config: Dict[str, Any]):
        """Save model summary and configuration"""
        summary_path = os.path.join(self.model_checkpoint_dir, "model_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Created: {datetime.now().isoformat()}\n")
            f.write("=" * 50 + "\n\n")
            
            # Model architecture
            f.write("Model Architecture:\n")
            f.write(str(model))
            f.write("\n\n")
            
            # Model parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            f.write(f"Total Parameters: {total_params:,}\n")
            f.write(f"Trainable Parameters: {trainable_params:,}\n\n")
            
            # Configuration
            f.write("Configuration:\n")
            f.write(json.dumps(config, indent=2))
        
        # Save configuration as JSON
        config_path = os.path.join(self.model_checkpoint_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def export_model(self, model: torch.nn.Module, export_path: str, 
                    format: str = 'pytorch'):
        """
        Export model in different formats
        
        Args:
            model: Model to export
            export_path: Path to save exported model
            format: Export format ('pytorch', 'onnx', 'torchscript')
        """
        if format.lower() == 'pytorch':
            torch.save(model.state_dict(), export_path)
        elif format.lower() == 'torchscript':
            model.eval()
            example_input = torch.randn(1, 4, 128, 128, 128)  # Adjust based on your input size
            traced_model = torch.jit.trace(model, example_input)
            traced_model.save(export_path)
        elif format.lower() == 'onnx':
            model.eval()
            example_input = torch.randn(1, 4, 128, 128, 128)  # Adjust based on your input size
            torch.onnx.export(
                model, example_input, export_path,
                export_params=True, opset_version=11,
                do_constant_folding=True,
                input_names=['input'], output_names=['output']
            )
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Exported model to: {export_path}")

class CheckpointManagerFactory:
    """Factory for creating checkpoint managers"""
    
    @staticmethod
    def create_manager(model_name: str, checkpoint_dir: str = "checkpoints", **kwargs) -> CheckpointManager:
        """Create a checkpoint manager"""
        return CheckpointManager(checkpoint_dir, model_name, **kwargs)
    
    @staticmethod
    def create_managers_for_cv(model_name: str, n_folds: int, 
                              checkpoint_dir: str = "checkpoints", **kwargs) -> List[CheckpointManager]:
        """Create checkpoint managers for cross-validation"""
        managers = []
        for fold in range(n_folds):
            fold_model_name = f"{model_name}_fold_{fold}"
            manager = CheckpointManager(checkpoint_dir, fold_model_name, **kwargs)
            managers.append(manager)
        return managers

# Example usage
if __name__ == "__main__":
    # Test checkpoint manager
    import torch.nn as nn
    
    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(4, 3, 3, padding=1)
        
        def forward(self, x):
            return self.conv(x)
    
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Create checkpoint manager
    manager = CheckpointManagerFactory.create_manager("test_model", "test_checkpoints")
    
    # Save model summary
    config = {"learning_rate": 1e-4, "batch_size": 2}
    manager.save_model_summary(model, config)
    
    # Simulate training
    for epoch in range(5):
        # Simulate metrics
        metrics = {"dice_score": 0.8 + epoch * 0.02, "loss": 0.5 - epoch * 0.05}
        loss = 0.5 - epoch * 0.05
        
        # Check if this is the best model
        is_best = manager.update_best_metric(metrics["dice_score"], epoch)
        
        # Save checkpoint
        checkpoint_path = manager.save_checkpoint(
            epoch, model, optimizer, None, metrics, loss, is_best
        )
        print(f"Saved checkpoint: {checkpoint_path}")
    
    # List checkpoints
    checkpoints = manager.list_checkpoints()
    print(f"\nTotal checkpoints: {len(checkpoints)}")
    
    # Load best checkpoint
    best_checkpoint = manager.load_best_checkpoint(model, optimizer)
    print(f"Loaded best checkpoint from epoch {best_checkpoint['epoch']}")
    
    # Export model
    manager.export_model(model, "test_model_export.pth", "pytorch")
    print("Model exported successfully")
