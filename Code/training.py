import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from tqdm.notebook import tqdm

from monai.networks.nets import AttentionUnet
from monai.transforms import Compose, RandFlipd, RandRotate90d, Rand3DElasticd, RandScaleIntensityd



BASE_DIR = './BRATS'
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')

BEST_MODEL_PATH = f'./models/best_model_fold_0_newsba0.pth'
BEST_MODEL_PATH = f'./models/best_model_fold_0_newsba1.pth'
BEST_MODEL_PATH = f'./models/best_model_fold_0_newsba2.pth'
BEST_MODEL_PATH = f'./models/best_model_fold_0_newsba3.pth'
BEST_MODEL_PATH = f'./models/best_model_fold_0_newsba4.pth'

BEST_MODEL_PATH_save = f'./models/best_model_fold_0_newsb0.pth'
BEST_MODEL_PATH_save = f'./models/best_model_fold_0_newsb1.pth'
BEST_MODEL_PATH_save = f'./models/best_model_fold_0_newsb2.pth'
BEST_MODEL_PATH_save = f'./models/best_model_fold_0_newsba3.pth'
BEST_MODEL_PATH_save = f'./models/best_model_fold_0_newsba4.pth'



# --- Device Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Training Parameters ---
LEARNING_RATE = 1e-4  
ADDITIONAL_EPOCHS = 30 # SET HOW MANY MORE EPOCHS YOU WANT TO TRAIN
PATIENCE = 7
BATCH_SIZE = 2
NUM_WORKERS = 2
FOLD_TO_RUN = 0
RANDOM_STATE = 42
N_SPLITS = 5

isInit = True

# --- Dataset Class ---
class BraTSDataset(Dataset):
    def __init__(self, file_paths, augment=False):
        self.file_paths = file_paths
        self.augment = augment
        if self.augment:
            self.transform = Compose([
                RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2),
                RandRotate90d(keys=["image", "mask"], prob=0.5, max_k=3),
                RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                Rand3DElasticd(keys=["image", "mask"], sigma_range=(3, 5), magnitude_range=(10, 30),
                                prob=0.2, mode=('bilinear', 'nearest'), padding_mode='zeros')
            ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with np.load(self.file_paths[idx]) as data:
            image = torch.from_numpy(data['image'].astype(np.float32))
            mask = torch.from_numpy(data['mask'].astype(np.float32))
        if self.augment:
            data_dict = self.transform({"image": image, "mask": mask})
            return data_dict["image"], data_dict["mask"]
        return image, mask

# --- Loss Functions ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        logits, targets = logits.view(-1), targets.view(-1)
        intersection = (logits * targets).sum()
        sum_of_sets = logits.sum() + targets.sum()
        dice_coeff = (2. * intersection + self.smooth) / (sum_of_sets + self.smooth)
        return 1 - dice_coeff

class DiceBCELoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_bce=0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
    def forward(self, logits, targets):
        dice = self.dice_loss(torch.sigmoid(logits), targets)
        bce = self.bce_loss(logits, targets)
        return self.weight_dice * dice + self.weight_bce * bce

# --- Training and Validation Loops ---
def dice_score_per_class(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds) > 0.5
    dice_scores = []
    for i in range(preds.shape[1]):
        pred_flat = preds[:, i, ...].contiguous().view(-1)
        target_flat = targets[:, i, ...].contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice)
    return torch.stack(dice_scores)

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(loader, desc="Training", leave=False)
    for images, masks in progress_bar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
    return running_loss / len(loader)

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_dice_scores = []
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validating", leave=False):
            images, masks = images.to(device), masks.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            dice = dice_score_per_class(outputs.cpu(), masks.cpu())
            all_dice_scores.append(dice)
            running_loss += loss.item()
    
    avg_dice = torch.stack(all_dice_scores).mean(0)
    return running_loss / len(loader), avg_dice

# Data Loading
# --- Find and Split Files ---
VALID_FILES = sorted([os.path.join(PROCESSED_DIR, f) for f in os.listdir(PROCESSED_DIR) if f.endswith('.npz') and os.path.getsize(os.path.join(PROCESSED_DIR, f)) > 0])
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
train_indices, val_indices = list(kf.split(VALID_FILES))[FOLD_TO_RUN]
train_files = [VALID_FILES[i] for i in train_indices]
val_files = [VALID_FILES[i] for i in val_indices]

# --- Create Datasets and Dataloaders ---
train_dataset = BraTSDataset(file_paths=train_files, augment=True)
val_dataset = BraTSDataset(file_paths=val_files, augment=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=(NUM_WORKERS > 0))
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=(NUM_WORKERS > 0))

print(f"Data loaders created for Fold {FOLD_TO_RUN}. Training set: {len(train_files)}, Validation set: {len(val_files)}")

# Load Checkpoint and Resume Training Loop
# --- Initialize Model, Optimizer, and Scheduler ---
model = AttentionUnet(
    spatial_dims=3, in_channels=4, out_channels=3,
    channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
).to(device)

criterion = DiceBCELoss().to(device)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler()

# --- Load Checkpoint ---
if os.path.exists(BEST_MODEL_PATH):
    print(f"Loading checkpoint from {BEST_MODEL_PATH}...")
    checkpoint = torch.load(BEST_MODEL_PATH)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_val_dice = checkpoint['best_val_dice']
    no_improve_epochs = 0 # Reset patience counter
    
    print(f"Resuming training from epoch {start_epoch}. Best validation Dice so far: {best_val_dice:.4f}")
else:
    print("No checkpoint found. Starting training from scratch.")
    start_epoch = 0
    best_val_dice = -1.0
    no_improve_epochs = 0

# The scheduler needs to be re-initialized based on the total number of epochs
TOTAL_EPOCHS = start_epoch + ADDITIONAL_EPOCHS
poly_lambda = lambda epoch: (1 - epoch / TOTAL_EPOCHS) ** 0.9
scheduler = LambdaLR(optimizer, lr_lambda=poly_lambda)

# --- Main Training Loop ---
start_time = time.time()
print(f"\n🚀 Resuming Training for {ADDITIONAL_EPOCHS} more epochs...")

for epoch in range(start_epoch, TOTAL_EPOCHS):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
    val_loss, val_dice_per_class = validate_one_epoch(model, val_loader, criterion, device)
    scheduler.step()
    
    avg_val_dice = val_dice_per_class.mean().item()
    
    print(f"Epoch {epoch+1}/{TOTAL_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Dice (Avg): {avg_val_dice:.4f} [TC: {val_dice_per_class[0]:.4f}, WT: {val_dice_per_class[1]:.4f}, ET: {val_dice_per_class[2]:.4f}]")
    
    if avg_val_dice > best_val_dice or isInit:
        best_val_dice = avg_val_dice
        no_improve_epochs = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_dice': best_val_dice
        }, BEST_MODEL_PATH_save)
        print(f"✨ New best model saved (Dice: {best_val_dice:.4f})")
        isInit = False
    else:
        no_improve_epochs += 1
    
    if no_improve_epochs >= PATIENCE:
        print(f"Early stopping at epoch {epoch+1} (no improvement in {PATIENCE} epochs).")
        break

print(f"\n✅ Training Finished in {(time.time() - start_time) / 60:.2f} minutes.")