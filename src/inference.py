"""
Inference pipeline for brain MRI segmentation
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from typing import Dict, List, Tuple, Optional, Union
import logging
from tqdm import tqdm
import json
from datetime import datetime

from config.config import Config
from src.models import create_model
from src.preprocessing import BraTS2023Preprocessor
from src.metrics import SegmentationEvaluator
from src.checkpoints import CheckpointManager

class InferenceEngine:
    """
    Inference engine for brain MRI segmentation
    """
    
    def __init__(self, config: Config, model_path: str, device: str = 'cuda'):
        self.config = config
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.model = None
        self.preprocessor = None
        self.metrics_computer = None
        
        # Load model
        self.load_model()
    
    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger("InferenceEngine")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def load_model(self):
        """Load trained model"""
        self.logger.info(f"Loading model from: {self.model_path}")
        
        # Create model
        self.model = create_model(
            self.config.model.model_name,
            in_channels=self.config.model.in_channels,
            out_channels=self.config.model.out_channels,
            features=self.config.model.features,
            dropout=0.0  # No dropout during inference
        ).to(self.device)
        
        # Load model weights
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.logger.info("Model loaded successfully")
        else:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Set to evaluation mode
        self.model.eval()
        
        # Initialize preprocessor
        self.preprocessor = BraTS2023Preprocessor(self.config.data)
        
        # Initialize metrics computer
        self.metrics_computer = SegmentationEvaluator(
            num_classes=self.config.model.out_channels,
            include_background=False
        )
    
    def preprocess_single_volume(self, volume_paths: Dict[str, str]) -> torch.Tensor:
        """
        Preprocess a single volume for inference
        
        Args:
            volume_paths: Dictionary with modality paths
            
        Returns:
            Preprocessed volume tensor
        """
        # Create data dictionary
        data_dict = {key: volume_paths[key] for key in self.config.data.modality_keys}
        
        # Get validation transforms (no augmentation)
        transforms = self.preprocessor.get_val_transforms()
        
        # Apply transforms
        data = transforms(data_dict)
        
        # Stack modalities
        modalities = []
        for modality in self.config.data.modality_keys:
            modalities.append(data[modality])
        
        # Stack along channel dimension
        volume = torch.stack(modalities, dim=0).unsqueeze(0)  # Add batch dimension
        
        return volume.to(self.device)
    
    def predict_single_volume(self, volume: torch.Tensor, 
                             use_tta: bool = False) -> torch.Tensor:
        """
        Predict segmentation for a single volume
        
        Args:
            volume: Input volume tensor
            use_tta: Whether to use test-time augmentation
            
        Returns:
            Predicted segmentation tensor
        """
        with torch.no_grad():
            if use_tta:
                # Test-time augmentation
                predictions = []
                
                # Original
                pred = self.model(volume)
                predictions.append(pred)
                
                # Flip along different axes
                for axis in [2, 3, 4]:  # Spatial axes
                    flipped_volume = torch.flip(volume, dims=[axis])
                    flipped_pred = self.model(flipped_volume)
                    flipped_pred = torch.flip(flipped_pred, dims=[axis])
                    predictions.append(flipped_pred)
                
                # Average predictions
                prediction = torch.mean(torch.stack(predictions), dim=0)
            else:
                prediction = self.model(volume)
            
            # Apply softmax to get probabilities
            prediction = F.softmax(prediction, dim=1)
            
            return prediction
    
    def postprocess_prediction(self, prediction: torch.Tensor, 
                              original_shape: Tuple[int, ...],
                              threshold: float = 0.5) -> np.ndarray:
        """
        Postprocess prediction
        
        Args:
            prediction: Model prediction tensor
            original_shape: Original volume shape
            threshold: Threshold for binary segmentation
            
        Returns:
            Postprocessed segmentation array
        """
        # Convert to numpy
        prediction = prediction.cpu().numpy()
        
        # Get argmax for class prediction
        segmentation = np.argmax(prediction, axis=1)[0]  # Remove batch dimension
        
        # Resize to original shape if needed
        if segmentation.shape != original_shape:
            segmentation = self._resize_segmentation(segmentation, original_shape)
        
        return segmentation
    
    def _resize_segmentation(self, segmentation: np.ndarray, 
                            target_shape: Tuple[int, ...]) -> np.ndarray:
        """Resize segmentation to target shape"""
        # Use nearest neighbor interpolation for segmentation
        from scipy.ndimage import zoom
        
        zoom_factors = [target_shape[i] / segmentation.shape[i] for i in range(len(target_shape))]
        resized_segmentation = zoom(segmentation, zoom_factors, order=0)
        
        return resized_segmentation
    
    def predict_volume(self, volume_paths: Dict[str, str], 
                      use_tta: bool = False) -> np.ndarray:
        """
        Predict segmentation for a volume from file paths
        
        Args:
            volume_paths: Dictionary with modality paths
            use_tta: Whether to use test-time augmentation
            
        Returns:
            Predicted segmentation array
        """
        # Load original shape for postprocessing
        original_shape = self._get_volume_shape(volume_paths[self.config.data.modality_keys[0]])
        
        # Preprocess
        volume = self.preprocess_single_volume(volume_paths)
        
        # Predict
        prediction = self.predict_single_volume(volume, use_tta)
        
        # Postprocess
        segmentation = self.postprocess_prediction(prediction, original_shape)
        
        return segmentation
    
    def _get_volume_shape(self, volume_path: str) -> Tuple[int, ...]:
        """Get original volume shape"""
        try:
            volume = nib.load(volume_path)
            return volume.shape
        except Exception as e:
            self.logger.warning(f"Could not load volume shape from {volume_path}: {e}")
            return (128, 128, 128)  # Default shape
    
    def predict_batch(self, volume_paths_list: List[Dict[str, str]], 
                     use_tta: bool = False) -> List[np.ndarray]:
        """
        Predict segmentation for a batch of volumes
        
        Args:
            volume_paths_list: List of dictionaries with modality paths
            use_tta: Whether to use test-time augmentation
            
        Returns:
            List of predicted segmentation arrays
        """
        predictions = []
        
        for volume_paths in tqdm(volume_paths_list, desc="Predicting"):
            try:
                prediction = self.predict_volume(volume_paths, use_tta)
                predictions.append(prediction)
            except Exception as e:
                self.logger.error(f"Error predicting volume: {e}")
                predictions.append(None)
        
        return predictions
    
    def evaluate_volume(self, volume_paths: Dict[str, str], 
                       ground_truth_path: str) -> Dict[str, float]:
        """
        Evaluate prediction against ground truth
        
        Args:
            volume_paths: Dictionary with modality paths
            ground_truth_path: Path to ground truth segmentation
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Predict
        prediction = self.predict_volume(volume_paths)
        
        # Load ground truth
        gt_volume = nib.load(ground_truth_path)
        ground_truth = gt_volume.get_fdata().astype(np.int32)
        
        # Convert to tensors for metrics computation
        pred_tensor = torch.from_numpy(prediction).unsqueeze(0).unsqueeze(0)
        gt_tensor = torch.from_numpy(ground_truth).unsqueeze(0)
        
        # Compute metrics
        metrics = self.metrics_computer.compute_all_metrics(pred_tensor, gt_tensor)
        
        return metrics
    
    def save_prediction(self, prediction: np.ndarray, output_path: str, 
                       reference_path: Optional[str] = None):
        """
        Save prediction as NIfTI file
        
        Args:
            prediction: Predicted segmentation array
            output_path: Path to save prediction
            reference_path: Path to reference volume for header information
        """
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if reference_path and os.path.exists(reference_path):
            # Use reference volume for header information
            reference_volume = nib.load(reference_path)
            prediction_volume = nib.Nifti1Image(
                prediction.astype(np.int32),
                reference_volume.affine,
                reference_volume.header
            )
        else:
            # Create new volume
            prediction_volume = nib.Nifti1Image(
                prediction.astype(np.int32),
                np.eye(4)
            )
        
        # Save
        nib.save(prediction_volume, output_path)
        self.logger.info(f"Prediction saved to: {output_path}")

class InferencePipeline:
    """
    Complete inference pipeline for batch processing
    """
    
    def __init__(self, config: Config, model_path: str, output_dir: str):
        self.config = config
        self.model_path = model_path
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize inference engine
        self.inference_engine = InferenceEngine(config, model_path)
        
        # Setup logging
        self.logger = logging.getLogger("InferencePipeline")
    
    def process_dataset(self, data_path: str, use_tta: bool = False, 
                       save_predictions: bool = True) -> Dict[str, List[float]]:
        """
        Process entire dataset
        
        Args:
            data_path: Path to dataset
            use_tta: Whether to use test-time augmentation
            save_predictions: Whether to save predictions
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Get all patient directories
        patient_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        
        all_metrics = []
        predictions_dir = os.path.join(self.output_dir, "predictions")
        
        if save_predictions:
            os.makedirs(predictions_dir, exist_ok=True)
        
        for patient_id in tqdm(patient_dirs, desc="Processing patients"):
            try:
                patient_path = os.path.join(data_path, patient_id)
                
                # Find modality files
                volume_paths = {}
                found_all_modalities = True
                
                for modality in self.config.data.modality_keys:
                    modality_file = os.path.join(patient_path, f"{patient_id}_{modality}.nii.gz")
                    if os.path.exists(modality_file):
                        volume_paths[modality] = modality_file
                    else:
                        found_all_modalities = False
                        break
                
                if not found_all_modalities:
                    self.logger.warning(f"Missing modalities for patient {patient_id}")
                    continue
                
                # Find ground truth
                gt_file = os.path.join(patient_path, f"{patient_id}_seg.nii.gz")
                if not os.path.exists(gt_file):
                    self.logger.warning(f"No ground truth for patient {patient_id}")
                    continue
                
                # Evaluate
                metrics = self.inference_engine.evaluate_volume(volume_paths, gt_file)
                all_metrics.append(metrics)
                
                # Save prediction if requested
                if save_predictions:
                    prediction = self.inference_engine.predict_volume(volume_paths, use_tta)
                    output_path = os.path.join(predictions_dir, f"{patient_id}_pred.nii.gz")
                    self.inference_engine.save_prediction(
                        prediction, output_path, volume_paths[self.config.data.modality_keys[0]]
                    )
                
            except Exception as e:
                self.logger.error(f"Error processing patient {patient_id}: {e}")
                continue
        
        # Compute average metrics
        if all_metrics:
            avg_metrics = self._compute_average_metrics(all_metrics)
            
            # Save results
            results_path = os.path.join(self.output_dir, "evaluation_results.json")
            with open(results_path, 'w') as f:
                json.dump({
                    'individual_metrics': all_metrics,
                    'average_metrics': avg_metrics,
                    'num_patients': len(all_metrics)
                }, f, indent=2)
            
            self.logger.info(f"Evaluation completed. Results saved to: {results_path}")
            return avg_metrics
        else:
            self.logger.error("No valid predictions generated")
            return {}
    
    def _compute_average_metrics(self, all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Compute average metrics across all patients"""
        if not all_metrics:
            return {}
        
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [metrics[key] for metrics in all_metrics]
            avg_metrics[key] = np.mean(values)
            avg_metrics[f"{key}_std"] = np.std(values)
        
        return avg_metrics

# Example usage
if __name__ == "__main__":
    from config.config import get_config
    
    # Get configuration
    config = get_config('unet3d')
    
    # Create inference pipeline
    model_path = "checkpoints/unet3d_fold_0/best_model_epoch_50.pth"
    output_dir = "outputs/inference_results"
    
    pipeline = InferencePipeline(config, model_path, output_dir)
    
    # Process dataset
    data_path = "data/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"
    results = pipeline.process_dataset(data_path, use_tta=True, save_predictions=True)
    
    print("Evaluation Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
