# Cross-Validation and K-Fold Explained

## 🎯 What is Cross-Validation?

**Cross-validation** is a technique used to evaluate machine learning models by splitting the dataset into multiple parts, training on some parts, and testing on others. It helps ensure that your model performs well on unseen data and isn't just memorizing the training data.

## 📊 K-Fold Cross-Validation

**K-Fold Cross-Validation** is a specific type of cross-validation where the dataset is divided into **K equal parts (folds)**. The model is trained K times, each time using K-1 folds for training and 1 fold for validation.

### 🔢 Your Project Configuration

```python
n_folds: int = 5  # 5-fold cross-validation
random_seed: int = 42  # For reproducible splits
```

## 📈 How 5-Fold Cross-Validation Works

### Dataset Split Example (1,251 patients):

```
Total Dataset: 1,251 patients
├── Fold 0: 250 patients (20%)
├── Fold 1: 250 patients (20%)  
├── Fold 2: 250 patients (20%)
├── Fold 3: 250 patients (20%)
└── Fold 4: 251 patients (20%)
```

### Training Process:

```
Iteration 1: Train on Folds 1,2,3,4 (1,001 patients) → Validate on Fold 0 (250 patients)
Iteration 2: Train on Folds 0,2,3,4 (1,001 patients) → Validate on Fold 1 (250 patients)
Iteration 3: Train on Folds 0,1,3,4 (1,001 patients) → Validate on Fold 2 (250 patients)
Iteration 4: Train on Folds 0,1,2,4 (1,001 patients) → Validate on Fold 3 (250 patients)
Iteration 5: Train on Folds 0,1,2,3 (1,000 patients) → Validate on Fold 4 (251 patients)
```

## 🏗️ Implementation in Your Project

### 1. Data Splitting (`src/preprocessing.py`):

```python
def create_cross_validation_splits(self, data_dicts):
    """Create cross-validation splits"""
    kfold = KFold(n_splits=self.config.n_folds, shuffle=True, random_state=self.config.random_seed)
    splits = []
    
    indices = np.arange(len(data_dicts))
    for train_idx, val_idx in kfold.split(indices):
        train_data = [data_dicts[i] for i in train_idx]
        val_data = [data_dicts[i] for i in val_idx]
        splits.append((train_data, val_data))
    
    return splits
```

### 2. Training Loop (`src/train.py`):

```python
class CrossValidationTrainer:
    def train_all_folds(self):
        for fold in range(self.config.data.n_folds):  # 0, 1, 2, 3, 4
            # Create trainer for this fold
            trainer = Trainer(self.config, fold)
            
            # Train on this fold
            trainer.train()
            
            # Store results
            self.fold_results.append({
                'fold': fold,
                'best_metric': trainer.best_metric,
                'training_history': trainer.training_history
            })
```

## 📁 File Structure During Cross-Validation

```
checkpoints/
├── unet3d_fold_0/
│   ├── best_model_epoch_15.pth
│   ├── checkpoint_epoch_5.pth
│   └── checkpoint_epoch_10.pth
├── unet3d_fold_1/
│   ├── best_model_epoch_23.pth
│   └── checkpoint_epoch_5.pth
├── unet3d_fold_2/
│   ├── best_model_epoch_18.pth
│   └── checkpoint_epoch_5.pth
├── unet3d_fold_3/
│   ├── best_model_epoch_21.pth
│   └── checkpoint_epoch_5.pth
└── unet3d_fold_4/
    ├── best_model_epoch_19.pth
    └── checkpoint_epoch_5.pth
```

## 📊 Benefits of Cross-Validation

### 1. **Robust Evaluation**:
- Tests model on different data splits
- Reduces overfitting to specific data
- More reliable performance estimates

### 2. **Better Generalization**:
- Model learns from diverse data combinations
- Less likely to memorize training data
- Better performance on unseen data

### 3. **Statistical Significance**:
- Multiple performance measurements
- Can calculate mean and standard deviation
- More confident in results

## 🎯 How to Use Cross-Validation in Your Project

### Command Line Usage:

```bash
# Train with 5-fold cross-validation
python main.py --mode cv --model unet3d --n_folds 5 --epochs 100

# Train single fold (e.g., fold 2)
python main.py --mode train --model unet3d --fold 2 --epochs 100

# Resume from specific fold
python main.py --mode cv --model unet3d --resume_fold 3
```

### Programmatic Usage:

```python
from src.train import CrossValidationTrainer
from config.config import get_config

# Get configuration
config = get_config('unet3d')

# Create cross-validation trainer
cv_trainer = CrossValidationTrainer(config)

# Train all folds
cv_trainer.train_all_folds()

# Get results
for fold_result in cv_trainer.fold_results:
    print(f"Fold {fold_result['fold']}: Best Dice = {fold_result['best_metric']:.4f}")
```

## 📈 Results Analysis

### After Cross-Validation Training:

```python
# Calculate average performance
dice_scores = [result['best_metric'] for result in cv_trainer.fold_results]
mean_dice = np.mean(dice_scores)
std_dice = np.std(dice_scores)

print(f"Cross-Validation Results:")
print(f"Mean Dice Score: {mean_dice:.4f} ± {std_dice:.4f}")
print(f"Best Fold: {np.argmax(dice_scores)} (Dice: {max(dice_scores):.4f})")
print(f"Worst Fold: {np.argmin(dice_scores)} (Dice: {min(dice_scores):.4f})")
```

## ⚡ Advantages for Medical Imaging

### 1. **Limited Data**:
- Medical datasets are often small
- Cross-validation maximizes data usage
- Every patient used for both training and validation

### 2. **Patient Diversity**:
- Different patients in each fold
- Tests generalization across patient populations
- Reduces patient-specific overfitting

### 3. **Robust Metrics**:
- Multiple performance measurements
- Statistical confidence in results
- Better comparison between models

## 🎯 Best Practices

### 1. **Stratified Splits** (Optional Enhancement):
```python
# Ensure each fold has similar class distribution
from sklearn.model_selection import StratifiedKFold
```

### 2. **Consistent Preprocessing**:
- Same preprocessing for all folds
- No data leakage between folds
- Reproducible results

### 3. **Model Selection**:
- Use average performance across folds
- Consider standard deviation
- Choose model with best generalization

## 🚀 Your Project's Cross-Validation Features

✅ **5-Fold Cross-Validation**: Default configuration
✅ **Reproducible Splits**: Fixed random seed (42)
✅ **Automatic Training**: Train all folds sequentially
✅ **Individual Fold Results**: Separate checkpoints per fold
✅ **Resume Capability**: Resume from specific fold
✅ **Comprehensive Logging**: Detailed logs per fold

**Cross-validation ensures your brain MRI segmentation model is robust and generalizes well to new patients!** 🎉
