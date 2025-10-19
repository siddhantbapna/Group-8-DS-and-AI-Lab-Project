@echo off
echo Starting BraTS2023 Data Preprocessing and Training
echo ==================================================

echo.
echo Step 1: Preprocessing data...
echo -----------------------------
pika\Scripts\python.exe preprocess_data.py --model unet3d

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Preprocessing failed!
    pause
    exit /b 1
)

echo.
echo Step 2: Training with preprocessed data...
echo ------------------------------------------
pika\Scripts\python.exe src\train_preprocessed.py --model unet3d --epochs 100 --batch_size 2 --use_preprocessed

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Training failed!
    pause
    exit /b 1
)

echo.
echo SUCCESS: Preprocessing and training completed!
echo ==============================================
pause
