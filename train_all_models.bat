@echo off
echo ========================================
echo Brain MRI Segmentation - Train All Models
echo ========================================
echo.

echo Available options:
echo 1. Quick test (5 epochs each, 2 folds)
echo 2. Full training (100 epochs each, 5 folds)
echo 3. Custom training
echo.

set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo Starting quick test training...
    echo This will train all models with 5 epochs each
    echo.
    pika\Scripts\python.exe quick_train_all.py
) else if "%choice%"=="2" (
    echo.
    echo Starting full training...
    echo This will train all models with cross-validation
    echo Estimated time: 10-15 hours
    echo.
    pika\Scripts\python.exe train_all_models.py
) else if "%choice%"=="3" (
    echo.
    echo Custom training options:
    echo.
    set /p models="Enter models to train (space-separated, or 'all'): "
    set /p epochs="Enter number of epochs: "
    set /p cv="Use cross-validation? (y/n): "
    
    if "%cv%"=="y" (
        pika\Scripts\python.exe train_all_models.py --models %models% --epochs %epochs%
    ) else (
        pika\Scripts\python.exe train_all_models.py --models %models% --epochs %epochs% --no-cv
    )
) else (
    echo Invalid choice. Please run the script again.
)

echo.
echo Training completed!
pause
