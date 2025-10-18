# Metrics Documentation for Brain MRI Segmentation

## Overview
This document provides comprehensive documentation of all metrics implemented in our BraTS2023 brain MRI segmentation project, including their mathematical formulations, implementations, interpretations, and use cases for evaluating model performance.

## Available Metrics

### 1. Dice Score (Sørensen-Dice Coefficient)

#### **Mathematical Formulation:**
```
Dice = (2 * |A ∩ B|) / (|A| + |B|)
```
Where:
- A = Predicted segmentation
- B = Ground truth segmentation
- |A ∩ B| = Intersection of predicted and ground truth
- |A|, |B| = Cardinality of predicted and ground truth sets

#### **Range:** 0 to 1 (higher is better)
- **0:** No overlap between prediction and ground truth
- **1:** Perfect overlap between prediction and ground truth

#### **Implementation:**
```python
from monai.metrics import DiceMetric

def compute_dice_score(y_pred, y_true, include_background=False):
    """
    Compute Dice score for segmentation
    
    Args:
        y_pred: Predicted segmentation (B, C, D, H, W)
        y_true: Ground truth segmentation (B, C, D, H, W)
        include_background: Whether to include background class
    
    Returns:
        dice_scores: Dice scores for each class
    """
    dice_metric = DiceMetric(include_background=include_background, reduction="mean")
    dice_scores = dice_metric(y_pred, y_true)
    return dice_scores
```

#### **Advantages:**
- ✅ Directly measures segmentation overlap
- ✅ Handles class imbalance naturally
- ✅ Standard metric for medical segmentation
- ✅ Intuitive interpretation
- ✅ Robust to object size

#### **Disadvantages:**
- ❌ May not penalize boundary errors heavily
- ❌ Can be misleading with very small objects
- ❌ Doesn't distinguish between false positives and false negatives

#### **Best For:**
- Medical image segmentation
- Imbalanced datasets
- When overlap is the primary concern
- Standard evaluation metric

---

### 2. Accuracy Metrics

#### **2.1 Pixel-wise Accuracy**
**Mathematical Formulation:**
```
Pixel Accuracy = (Correct Pixels) / (Total Pixels)
```

**Implementation:**
```python
def compute_pixel_accuracy(y_pred, y_true):
    """
    Compute pixel-wise accuracy
    
    Args:
        y_pred: Predicted segmentation (B, C, D, H, W)
        y_true: Ground truth segmentation (B, C, D, H, W)
    
    Returns:
        accuracy: Pixel-wise accuracy
    """
    if y_true.dim() == 5 and y_true.shape[1] > 1:  # One-hot encoded
        target_classes = torch.argmax(y_true, dim=1)
    else:
        target_classes = y_true.long()
    
    pred_classes = torch.argmax(y_pred, dim=1)
    correct = (pred_classes == target_classes).float()
    accuracy = correct.mean()
    return accuracy
```

#### **2.2 Class-balanced Accuracy**
**Mathematical Formulation:**
```
Balanced Accuracy = (1/C) * Σ(TP_i / (TP_i + FN_i))
```
Where C is the number of classes.

**Implementation:**
```python
def compute_balanced_accuracy(y_pred, y_true, num_classes):
    """
    Compute class-balanced accuracy
    
    Args:
        y_pred: Predicted segmentation (B, C, D, H, W)
        y_true: Ground truth segmentation (B, C, D, H, W)
        num_classes: Number of classes
    
    Returns:
        balanced_accuracy: Class-balanced accuracy
    """
    if y_true.dim() == 5 and y_true.shape[1] > 1:  # One-hot encoded
        target_classes = torch.argmax(y_true, dim=1)
    else:
        target_classes = y_true.long()
    
    pred_classes = torch.argmax(y_pred, dim=1)
    
    class_accuracies = []
    for class_id in range(num_classes):
        class_mask = (target_classes == class_id)
        if class_mask.sum() > 0:
            class_correct = (pred_classes == target_classes) & class_mask
            class_accuracy = class_correct.sum().float() / class_mask.sum().float()
            class_accuracies.append(class_accuracy)
    
    balanced_accuracy = torch.stack(class_accuracies).mean()
    return balanced_accuracy
```

#### **Advantages:**
- ✅ Simple and intuitive
- ✅ Easy to interpret
- ✅ Good for balanced datasets
- ✅ Fast computation

#### **Disadvantages:**
- ❌ Can be misleading with imbalanced datasets
- ❌ Doesn't account for class importance
- ❌ May not reflect segmentation quality

#### **Best For:**
- Balanced datasets
- Quick evaluation
- Baseline comparisons
- When pixel-level accuracy matters

---

### 3. Hausdorff Distance

#### **Mathematical Formulation:**
```
Hausdorff Distance = max(h(A,B), h(B,A))
```
Where:
```
h(A,B) = max(min(d(a,b))) for a∈A, b∈B
```

#### **Implementation:**
```python
from monai.metrics import HausdorffDistanceMetric

def compute_hausdorff_distance(y_pred, y_true, include_background=False):
    """
    Compute Hausdorff distance for segmentation boundaries
    
    Args:
        y_pred: Predicted segmentation (B, C, D, H, W)
        y_true: Ground truth segmentation (B, C, D, H, W)
        include_background: Whether to include background class
    
    Returns:
        hausdorff_distances: Hausdorff distances for each class
    """
    hausdorff_metric = HausdorffDistanceMetric(
        include_background=include_background, 
        reduction="mean",
        percentile=95  # Use 95th percentile for robustness
    )
    hausdorff_distances = hausdorff_metric(y_pred, y_true)
    return hausdorff_distances
```

#### **Advantages:**
- ✅ Measures boundary accuracy
- ✅ Robust to outliers
- ✅ Good for shape evaluation
- ✅ Important for medical applications

#### **Disadvantages:**
- ❌ Computationally expensive
- ❌ Sensitive to noise
- ❌ May not reflect overall segmentation quality
- ❌ Requires binary masks

#### **Best For:**
- Boundary-critical applications
- Medical image segmentation
- When shape accuracy is important
- Research and detailed analysis

---

### 4. Sensitivity (Recall)

#### **Mathematical Formulation:**
```
Sensitivity = TP / (TP + FN)
```
Where:
- TP = True Positives
- FN = False Negatives

#### **Implementation:**
```python
def compute_sensitivity(y_pred, y_true, class_id):
    """
    Compute sensitivity for a specific class
    
    Args:
        y_pred: Predicted segmentation (B, C, D, H, W)
        y_true: Ground truth segmentation (B, C, D, H, W)
        class_id: Class to compute sensitivity for
    
    Returns:
        sensitivity: Sensitivity for the specified class
    """
    if y_true.dim() == 5 and y_true.shape[1] > 1:  # One-hot encoded
        target_classes = torch.argmax(y_true, dim=1)
    else:
        target_classes = y_true.long()
    
    pred_classes = torch.argmax(y_pred, dim=1)
    
    # True positives: correctly predicted as class_id
    tp = ((pred_classes == class_id) & (target_classes == class_id)).sum().float()
    
    # False negatives: ground truth is class_id but predicted as other
    fn = ((pred_classes != class_id) & (target_classes == class_id)).sum().float()
    
    sensitivity = tp / (tp + fn + 1e-8)  # Add epsilon to avoid division by zero
    return sensitivity
```

#### **Advantages:**
- ✅ Measures detection rate
- ✅ Important for medical applications
- ✅ Good for imbalanced datasets
- ✅ Easy to interpret

#### **Disadvantages:**
- ❌ Doesn't account for false positives
- ❌ May not reflect overall performance
- ❌ Sensitive to threshold selection

#### **Best For:**
- Medical diagnosis
- When missing cases is critical
- Imbalanced datasets
- Detection tasks

---

### 5. Specificity

#### **Mathematical Formulation:**
```
Specificity = TN / (TN + FP)
```
Where:
- TN = True Negatives
- FP = False Positives

#### **Implementation:**
```python
def compute_specificity(y_pred, y_true, class_id):
    """
    Compute specificity for a specific class
    
    Args:
        y_pred: Predicted segmentation (B, C, D, H, W)
        y_true: Ground truth segmentation (B, C, D, H, W)
        class_id: Class to compute specificity for
    
    Returns:
        specificity: Specificity for the specified class
    """
    if y_true.dim() == 5 and y_true.shape[1] > 1:  # One-hot encoded
        target_classes = torch.argmax(y_true, dim=1)
    else:
        target_classes = y_true.long()
    
    pred_classes = torch.argmax(y_pred, dim=1)
    
    # True negatives: correctly predicted as not class_id
    tn = ((pred_classes != class_id) & (target_classes != class_id)).sum().float()
    
    # False positives: predicted as class_id but ground truth is other
    fp = ((pred_classes == class_id) & (target_classes != class_id)).sum().float()
    
    specificity = tn / (tn + fp + 1e-8)  # Add epsilon to avoid division by zero
    return specificity
```

#### **Advantages:**
- ✅ Measures true negative rate
- ✅ Important for medical applications
- ✅ Good for imbalanced datasets
- ✅ Easy to interpret

#### **Disadvantages:**
- ❌ Doesn't account for false negatives
- ❌ May not reflect overall performance
- ❌ Sensitive to threshold selection

#### **Best For:**
- Medical diagnosis
- When false positives are critical
- Imbalanced datasets
- Classification tasks

---

### 6. Precision

#### **Mathematical Formulation:**
```
Precision = TP / (TP + FP)
```

#### **Implementation:**
```python
def compute_precision(y_pred, y_true, class_id):
    """
    Compute precision for a specific class
    
    Args:
        y_pred: Predicted segmentation (B, C, D, H, W)
        y_true: Ground truth segmentation (B, C, D, H, W)
        class_id: Class to compute precision for
    
    Returns:
        precision: Precision for the specified class
    """
    if y_true.dim() == 5 and y_true.shape[1] > 1:  # One-hot encoded
        target_classes = torch.argmax(y_true, dim=1)
    else:
        target_classes = y_true.long()
    
    pred_classes = torch.argmax(y_pred, dim=1)
    
    # True positives: correctly predicted as class_id
    tp = ((pred_classes == class_id) & (target_classes == class_id)).sum().float()
    
    # False positives: predicted as class_id but ground truth is other
    fp = ((pred_classes == class_id) & (target_classes != class_id)).sum().float()
    
    precision = tp / (tp + fp + 1e-8)  # Add epsilon to avoid division by zero
    return precision
```

#### **Advantages:**
- ✅ Measures prediction accuracy
- ✅ Important for medical applications
- ✅ Good for imbalanced datasets
- ✅ Easy to interpret

#### **Disadvantages:**
- ❌ Doesn't account for false negatives
- ❌ May not reflect overall performance
- ❌ Sensitive to threshold selection

#### **Best For:**
- Medical diagnosis
- When false positives are critical
- Imbalanced datasets
- Classification tasks

---

### 7. F1 Score

#### **Mathematical Formulation:**
```
F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
```

#### **Implementation:**
```python
def compute_f1_score(y_pred, y_true, class_id):
    """
    Compute F1 score for a specific class
    
    Args:
        y_pred: Predicted segmentation (B, C, D, H, W)
        y_true: Ground truth segmentation (B, C, D, H, W)
        class_id: Class to compute F1 score for
    
    Returns:
        f1_score: F1 score for the specified class
    """
    precision = compute_precision(y_pred, y_true, class_id)
    recall = compute_sensitivity(y_pred, y_true, class_id)
    
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1_score
```

#### **Advantages:**
- ✅ Balances precision and recall
- ✅ Good for imbalanced datasets
- ✅ Single metric for evaluation
- ✅ Easy to interpret

#### **Disadvantages:**
- ❌ May not reflect overall performance
- ❌ Sensitive to threshold selection
- ❌ Doesn't account for true negatives

#### **Best For:**
- Imbalanced datasets
- When both precision and recall matter
- Classification tasks
- Balanced evaluation

---

## BraTS2023 Specific Metrics

### **Tumor Region Metrics:**
For BraTS2023, we evaluate three tumor regions:

1. **Whole Tumor (WT):** All tumor regions (classes 1, 2, 3)
2. **Tumor Core (TC):** Central tumor regions (classes 1, 3)
3. **Enhancing Tumor (ET):** Active tumor regions (class 3)

### **Implementation:**
```python
def compute_brats_metrics(y_pred, y_true):
    """
    Compute BraTS2023 specific metrics
    
    Args:
        y_pred: Predicted segmentation (B, 3, D, H, W)
        y_true: Ground truth segmentation (B, 3, D, H, W)
    
    Returns:
        metrics: Dictionary of BraTS2023 metrics
    """
    # Map BraTS labels: 0=Background, 1=NCR/NET, 2=ED, 3=ET
    # Our output: 0=Background, 1=NCR/NET/ED, 2=ET
    
    # Whole Tumor (WT): classes 1, 2, 3 -> our class 1
    wt_pred = (y_pred[:, 1:2, :, :, :] > 0.5).float()
    wt_true = (y_true[:, 1:2, :, :, :] > 0.5).float()
    wt_dice = compute_dice_score(wt_pred, wt_true)
    
    # Tumor Core (TC): classes 1, 3 -> our class 1 (NCR/NET) + class 2 (ET)
    tc_pred = (y_pred[:, 1:3, :, :, :].sum(dim=1, keepdim=True) > 0.5).float()
    tc_true = (y_true[:, 1:3, :, :, :].sum(dim=1, keepdim=True) > 0.5).float()
    tc_dice = compute_dice_score(tc_pred, tc_true)
    
    # Enhancing Tumor (ET): class 3 -> our class 2
    et_pred = (y_pred[:, 2:3, :, :, :] > 0.5).float()
    et_true = (y_true[:, 2:3, :, :, :] > 0.5).float()
    et_dice = compute_dice_score(et_pred, et_true)
    
    return {
        'dice_wt': wt_dice,
        'dice_tc': tc_dice,
        'dice_et': et_dice,
        'dice_mean': (wt_dice + tc_dice + et_dice) / 3
    }
```

---

## Metrics Comparison Table

| Metric | Range | Best For | Computational Cost | Medical Relevance |
|--------|-------|----------|-------------------|-------------------|
| **Dice Score** | 0-1 | Segmentation overlap | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Pixel Accuracy** | 0-1 | Overall correctness | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Balanced Accuracy** | 0-1 | Class-balanced evaluation | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Hausdorff Distance** | 0-∞ | Boundary accuracy | ⭐⭐ | ⭐⭐⭐⭐ |
| **Sensitivity** | 0-1 | Detection rate | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Specificity** | 0-1 | True negative rate | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Precision** | 0-1 | Prediction accuracy | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **F1 Score** | 0-1 | Balanced evaluation | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## Implementation in Our Project

### **MetricsComputer Class:**
```python
class MetricsComputer:
    """
    Comprehensive metrics computer for brain MRI segmentation
    """
    
    def __init__(self, num_classes=3, include_background=False):
        self.num_classes = num_classes
        self.include_background = include_background
        
        # Initialize metrics
        self.dice_metric = DiceMetric(include_background=include_background, reduction="mean")
        self.hausdorff_metric = HausdorffDistanceMetric(
            include_background=include_background, 
            reduction="mean",
            percentile=95
        )
    
    def compute_all_metrics(self, y_pred, y_true):
        """
        Compute all available metrics
        
        Args:
            y_pred: Predicted segmentation (B, C, D, H, W)
            y_true: Ground truth segmentation (B, C, D, H, W)
        
        Returns:
            metrics: Dictionary of all computed metrics
        """
        metrics = {}
        
        # Dice scores
        dice_scores = self.dice_metric(y_pred, y_true)
        metrics['dice_mean'] = dice_scores.mean().item()
        
        for i in range(self.num_classes):
            if not self.include_background and i == 0:
                continue
            metrics[f'dice_class_{i}'] = dice_scores[i].item()
        
        # Hausdorff distances
        try:
            hausdorff_distances = self.hausdorff_metric(y_pred, y_true)
            metrics['hausdorff_mean'] = hausdorff_distances.mean().item()
            
            for i in range(self.num_classes):
                if not self.include_background and i == 0:
                    continue
                metrics[f'hausdorff_class_{i}'] = hausdorff_distances[i].item()
        except Exception as e:
            # Hausdorff can fail with small objects
            metrics['hausdorff_mean'] = float('inf')
        
        # Accuracy metrics
        metrics['pixel_accuracy'] = self.compute_pixel_accuracy(y_pred, y_true)
        metrics['balanced_accuracy'] = self.compute_balanced_accuracy(y_pred, y_true)
        
        # BraTS2023 specific metrics
        brats_metrics = self.compute_brats_metrics(y_pred, y_true)
        metrics.update(brats_metrics)
        
        return metrics
```

### **Usage in Training:**
```python
# In src/train.py
def train_epoch(self):
    # ... training loop ...
    
    # Compute metrics
    with torch.no_grad():
        metrics = self.metrics_computer.compute_all_metrics(outputs, targets)
    
    # Log metrics
    self.logger.info(f"Dice: {metrics['dice_mean']:.4f}")
    self.logger.info(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    self.logger.info(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    
    # Log to TensorBoard
    self.writer.add_scalar('Metrics/Dice', metrics['dice_mean'], epoch)
    self.writer.add_scalar('Metrics/PixelAccuracy', metrics['pixel_accuracy'], epoch)
    self.writer.add_scalar('Metrics/BalancedAccuracy', metrics['balanced_accuracy'], epoch)
```

---

## Interpretation Guidelines

### **Dice Score Interpretation:**
- **0.9-1.0:** Excellent segmentation
- **0.8-0.9:** Good segmentation
- **0.7-0.8:** Fair segmentation
- **0.6-0.7:** Poor segmentation
- **<0.6:** Very poor segmentation

### **Accuracy Interpretation:**
- **>95%:** Excellent accuracy
- **90-95%:** Good accuracy
- **80-90%:** Fair accuracy
- **<80%:** Poor accuracy

### **Hausdorff Distance Interpretation:**
- **<1mm:** Excellent boundary accuracy
- **1-2mm:** Good boundary accuracy
- **2-5mm:** Fair boundary accuracy
- **>5mm:** Poor boundary accuracy

---

## Best Practices

### **1. Metric Selection:**
- Use **Dice Score** as primary metric for segmentation
- Use **Pixel Accuracy** for overall performance
- Use **Balanced Accuracy** for imbalanced datasets
- Use **Hausdorff Distance** for boundary-critical applications

### **2. Evaluation Strategy:**
- Compute metrics on validation set
- Use cross-validation for robust evaluation
- Monitor multiple metrics simultaneously
- Consider class-specific metrics

### **3. Reporting:**
- Report mean and standard deviation
- Include confidence intervals
- Show class-specific results
- Provide statistical significance tests

### **4. Monitoring:**
- Track metrics during training
- Use TensorBoard for visualization
- Set up early stopping based on metrics
- Monitor for overfitting

---

## Conclusion

The metrics implemented in our BraTS2023 project provide comprehensive evaluation of segmentation performance:

1. **Dice Score** - Primary metric for segmentation overlap
2. **Accuracy Metrics** - Overall and class-balanced performance
3. **Hausdorff Distance** - Boundary accuracy evaluation
4. **Clinical Metrics** - Sensitivity, specificity, precision, F1
5. **BraTS2023 Specific** - Tumor region evaluation

These metrics together provide a complete picture of model performance for medical image segmentation tasks, enabling informed decisions about model selection and deployment.
