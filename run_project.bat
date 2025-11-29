@echo off
setlocal

echo ========================================================
echo Secure Federated Learning Framework for Fraud Detection
echo ========================================================
echo.

set "DATA_DIR=data"
set "DATA_FILE=creditcard.csv"
set "FULL_DATA_PATH=%DATA_DIR%\%DATA_FILE%"

if not exist "%DATA_DIR%" (
    mkdir "%DATA_DIR%"
)

if exist "%DATA_DIR%\train_transaction.csv" (
    echo Found IEEE-CIS Fraud Detection dataset.
    echo Starting simulation...
    python run_simulation.py
) else if exist "%FULL_DATA_PATH%" (
    echo Found Credit Card Fraud dataset at %FULL_DATA_PATH%
    echo Starting simulation...
    python run_simulation.py --data-path "%FULL_DATA_PATH%"
) else (
    echo Dataset not found in %DATA_DIR%
    echo.
    echo Please ensure 'train_transaction.csv' is in the 'data' folder.
    echo Or enter the full path to your creditcard.csv file:
    set /p USER_DATA_PATH="> "
    
    if exist "%USER_DATA_PATH%" (
        echo.
        echo Starting simulation with provided dataset...
        python run_simulation.py --data-path "%USER_DATA_PATH%"
    ) else (
        echo.
        echo File not found. Running with synthetic data...
        python run_simulation.py
    )
)

echo.
echo Simulation complete.
pause
