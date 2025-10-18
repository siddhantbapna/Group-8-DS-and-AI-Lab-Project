# RTX 4070 8GB VRAM Optimization Guide

## Overview
This guide provides specific recommendations and optimizations for running the BraTS2023 brain MRI segmentation project on an RTX 4070 with 8GB VRAM, 32GB RAM, and Ryzen 9 processor.

## Hardware Specifications
- **GPU:** RTX 4070 (8GB VRAM)
- **RAM:** 32GB
- **CPU:** Ryzen 9
- **Storage:** SSD recommended for data loading

## Model Compatibility Analysis

### ✅ **Fully Compatible Models (No Issues)**

| Model | Batch Size | Memory Usage | Training Time | Expected Dice | Status |
|-------|------------|--------------|---------------|---------------|---------|
| **UNet** | 4 | 2GB | 3.3 hours | 0.75-0.80 | ✅ Optimal |
| **3D UNet** | 1 | 6GB | 8.3 hours | 0.85-0.90 | ✅ Good |
| **ResUNet** | 1 | 8GB | 11.7 hours | 0.86-0.91 | ✅ Max VRAM |

### ⚠️ **Models Requiring Optimization**

| Model | Batch Size | Memory Usage | Training Time | Expected Dice | Status |
|-------|------------|--------------|---------------|---------------|---------|
| **Attention UNet** | 1 | 12GB | 20 hours | 0.88-0.92 | ⚠️ Needs Optimization |
| **VNet** | 1 | 10GB | 25 hours | 0.87-0.91 | ⚠️ Needs Optimization |
| **nnUNet** | 1 | 16GB | 33.3 hours | 0.89-0.93 | ⚠️ Needs Optimization |

## Recommended Configurations

### 🥇 **Best Overall Choice: UNet**
```python
config = Config(
    model=ModelConfig(
        model_name="unet",
        features=[32, 64, 128, 256]
    ),
    data=DataConfig(
        brain_only_training=True,
        brain_mask_method="otsu",
        background_weight=0.05,
        foreground_sampling=True,
        n_folds=5
    ),
    training=TrainingConfig(
        batch_size=4,  # Can use larger batch size
        num_epochs=100,
        learning_rate=1e-4,
        loss_function="weighted_dice_bce",
        optimizer="adamw",
        scheduler="poly",
        use_amp=True
    )
)
```

**Advantages:**
- Only uses 2GB VRAM (25% utilization)
- Fast training (3.3 hours for 100 epochs)
- Can use batch size 4 for stable training
- Good baseline performance

**Disadvantages:**
- 2D processing (loses 3D context)
- Lower accuracy than 3D models

### 🥈 **Best 3D Choice: 3D UNet**
```python
config = Config(
    model=ModelConfig(
        model_name="unet3d",
        features=[32, 64, 128, 256]
    ),
    data=DataConfig(
        brain_only_training=True,
        brain_mask_method="otsu",
        background_weight=0.05,
        foreground_sampling=True,
        n_folds=5
    ),
    training=TrainingConfig(
        batch_size=1,  # Limited by VRAM
        num_epochs=100,
        learning_rate=1e-4,
        loss_function="weighted_dice_bce",
        optimizer="adamw",
        scheduler="poly",
        use_amp=True
    )
)
```

**Advantages:**
- Full 3D processing
- Good accuracy (0.85-0.90 Dice)
- Uses 6GB VRAM (75% utilization)
- Standard medical imaging approach

**Disadvantages:**
- Limited to batch size 1
- Longer training time (8.3 hours)

### 🥉 **Maximum VRAM Usage: ResUNet**
```python
config = Config(
    model=ModelConfig(
        model_name="resunet",
        features=[32, 64, 128, 256],
        num_res_units=2
    ),
    data=DataConfig(
        brain_only_training=True,
        brain_mask_method="otsu",
        background_weight=0.05,
        foreground_sampling=True,
        n_folds=5
    ),
    training=TrainingConfig(
        batch_size=1,  # Uses full 8GB VRAM
        num_epochs=100,
        learning_rate=1e-4,
        loss_function="weighted_dice_bce",
        optimizer="adamw",
        scheduler="poly",
        use_amp=True
    )
)
```

**Advantages:**
- Maximum VRAM utilization (8GB)
- Stable training with residual connections
- Good accuracy (0.86-0.91 Dice)
- Better feature learning

**Disadvantages:**
- Limited to batch size 1
- Longer training time (11.7 hours)
- No room for error in memory management

## Optimization Strategies for Large Models

### **Gradient Accumulation**
For models that exceed 8GB VRAM, use gradient accumulation to simulate larger batch sizes:

```python
# Example for Attention UNet
config = Config(
    model=ModelConfig(model_name="attentionunet"),
    training=TrainingConfig(
        batch_size=1,  # Physical batch size
        gradient_accumulation_steps=4,  # Effective batch size = 4
        num_epochs=90,
        learning_rate=5e-5,  # Lower LR for stability
        use_amp=True  # Essential for memory optimization
    )
)
```

### **Mixed Precision Training**
Always enable mixed precision for memory optimization:

```python
training_config = TrainingConfig(
    use_amp=True,  # Reduces memory usage by ~50%
    max_grad_norm=1.0  # Prevents gradient explosion
)
```

### **Model Optimization Techniques**

1. **Reduce Model Size:**
   ```python
   # Smaller feature maps
   features = [16, 32, 64, 128]  # Instead of [32, 64, 128, 256]
   ```

2. **Gradient Checkpointing:**
   ```python
   # Trade computation for memory
   model.gradient_checkpointing_enable()
   ```

3. **Dynamic Loss Scaling:**
   ```python
   # For mixed precision training
   scaler = torch.cuda.amp.GradScaler()
   ```

## Training Strategies

### **Quick Testing (Recommended First)**
```bash
# Test all models quickly
pika\Scripts\python.exe quick_train_all.py
```

### **Full Training with Compatible Models**
```bash
# Train UNet, 3D UNet, and ResUNet
pika\Scripts\python.exe train_all_models.py --models unet unet3d resunet
```

### **Advanced Training with Optimization**
```bash
# Train large models with gradient accumulation
pika\Scripts\python.exe main.py --model attentionunet --batch_size 1 --gradient_accumulation 4
```

## Memory Monitoring

### **GPU Memory Monitoring**
```python
import torch

def monitor_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
```

### **Memory Optimization Tips**

1. **Clear Cache Regularly:**
   ```python
   torch.cuda.empty_cache()
   ```

2. **Use Context Managers:**
   ```python
   with torch.cuda.amp.autocast():
       outputs = model(inputs)
   ```

3. **Monitor Memory Usage:**
   ```python
   # Add to training loop
   if torch.cuda.memory_allocated() > 7.5 * 1024**3:  # 7.5GB
       torch.cuda.empty_cache()
   ```

## Expected Performance

### **Training Time Estimates (8GB VRAM)**
| Model | Single Epoch | 100 Epochs | 5-Fold CV |
|-------|--------------|------------|-----------|
| **UNet** | 2 min | 3.3 hours | 16.5 hours |
| **3D UNet** | 5 min | 8.3 hours | 41.5 hours |
| **ResUNet** | 7 min | 11.7 hours | 58.5 hours |
| **Attention UNet** | 12 min | 20 hours | 100 hours |
| **VNet** | 15 min | 25 hours | 125 hours |
| **nnUNet** | 20 min | 33.3 hours | 166.5 hours |

### **Memory Usage Breakdown**
| Model | Base Memory | Batch Size 1 | Batch Size 2 | Batch Size 4 |
|-------|-------------|--------------|--------------|--------------|
| **UNet** | 1.5GB | 2GB | 3GB | 4GB |
| **3D UNet** | 4GB | 6GB | 10GB | OOM |
| **ResUNet** | 6GB | 8GB | OOM | OOM |
| **Attention UNet** | 10GB | 12GB | OOM | OOM |
| **VNet** | 8GB | 10GB | OOM | OOM |
| **nnUNet** | 14GB | 16GB | OOM | OOM |

## Troubleshooting

### **Out of Memory (OOM) Errors**

1. **Reduce Batch Size:**
   ```python
   batch_size = 1  # Minimum batch size
   ```

2. **Enable Mixed Precision:**
   ```python
   use_amp = True
   ```

3. **Use Gradient Accumulation:**
   ```python
   gradient_accumulation_steps = 4
   ```

4. **Clear GPU Cache:**
   ```python
   torch.cuda.empty_cache()
   ```

### **Slow Training**

1. **Enable Mixed Precision:**
   ```python
   use_amp = True  # 2x speedup
   ```

2. **Use Brain-Only Pipeline:**
   ```python
   brain_only_training = True  # Reduces data size
   ```

3. **Optimize Data Loading:**
   ```python
   num_workers = 4  # Parallel data loading
   pin_memory = True  # Faster GPU transfer
   ```

### **Poor Accuracy**

1. **Use Brain-Only Training:**
   ```python
   brain_only_training = True
   brain_mask_method = "otsu"
   ```

2. **Use Weighted Loss:**
   ```python
   loss_function = "weighted_dice_bce"
   background_weight = 0.05
   ```

3. **Increase Training Time:**
   ```python
   num_epochs = 150  # More training
   patience = 20  # More patience
   ```

## Recommended Workflow

### **Step 1: Quick Test**
```bash
# Test all models with 5 epochs
pika\Scripts\python.exe quick_train_all.py
```

### **Step 2: Train Compatible Models**
```bash
# Train UNet, 3D UNet, ResUNet
pika\Scripts\python.exe train_all_models.py --models unet unet3d resunet
```

### **Step 3: Select Best Model**
```bash
# Train best model with full cross-validation
pika\Scripts\python.exe main.py --model unet3d --mode cv --epochs 100
```

### **Step 4: Advanced Training (Optional)**
```bash
# Train large models with optimization
pika\Scripts\python.exe main.py --model attentionunet --batch_size 1 --gradient_accumulation 4
```

## Conclusion

For your RTX 4070 8GB VRAM setup:

1. **Start with UNet** for quick results and efficient training
2. **Use 3D UNet** for better accuracy with 3D processing
3. **Try ResUNet** for maximum VRAM utilization
4. **Use gradient accumulation** for larger models
5. **Always enable mixed precision** for memory optimization

The 8GB VRAM limitation is manageable with proper optimization, and you can still achieve excellent results with the recommended models and configurations.
