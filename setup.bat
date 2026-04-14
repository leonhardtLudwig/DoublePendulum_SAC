@echo off
REM =============================================================================
REM  Double Pendulum RL Project - Environment Setup (Windows)
REM  Leonardo Luigi Pepe, 2157562
REM
REM  Usage (from project root, where src\ is located):
REM    setup.bat
REM
REM  Requires: Anaconda or Miniconda installed and added to PATH
REM =============================================================================

setlocal enabledelayedexpansion

set CONDA_ENV_NAME=double_pendulum
set PYTHON_VERSION=3.10

echo.
echo ============================================================
echo    Double Pendulum RL Project - Setup Script (Windows)
echo ============================================================
echo.

REM -- Sanity check: must be run from project root
if not exist "src" (
    echo [X] 'src\' not found. Run this script from the project root.
    exit /b 1
)
echo [OK] Project root - src\ found.
echo.

REM -- STEP 1: Check conda
echo === STEP 1/3 - Python environment ===
echo.

where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo [X] conda not found.
    echo     Install Miniconda: https://docs.conda.io/en/latest/miniconda.html
    exit /b 1
)
echo [OK] conda found.
echo.

REM -- Check if env already exists
conda env list | find "%CONDA_ENV_NAME%" >nul 2>nul
if %errorlevel% equ 0 (
    echo [!] Conda env '%CONDA_ENV_NAME%' already exists - skipping creation.
) else (
    echo [*] Creating conda env '%CONDA_ENV_NAME%' with Python %PYTHON_VERSION%...
    call conda create -y -n %CONDA_ENV_NAME% python=%PYTHON_VERSION%
    if !errorlevel! neq 0 (
        echo [X] Failed to create conda environment.
        exit /b 1
    )
    echo [OK] Environment created.
)
echo.

REM -- Activate environment
call conda activate %CONDA_ENV_NAME%
if %errorlevel% neq 0 (
    echo [X] Failed to activate conda environment.
    exit /b 1
)
echo [OK] Activated: %CONDA_ENV_NAME%
echo.

REM -- Upgrade pip
python -m pip install --upgrade pip --quiet

REM -- STEP 2: Install double_pendulum from local src\
echo === STEP 2/3 - double_pendulum (local src\) ===
echo.

echo [*] Installing double_pendulum from .\src ...
pip install -q -e ".\src"
if %errorlevel% neq 0 (
    echo [X] Failed to install double_pendulum.
    exit /b 1
)
echo [OK] double_pendulum installed.
echo.

REM -- STEP 3: Install dependencies
echo === STEP 3/3 - Project dependencies ===
echo.

REM -- PyTorch (CUDA or CPU)
echo [*] Checking for NVIDIA GPU...
nvidia-smi >nul 2>nul
if %errorlevel% equ 0 (
    echo [OK] CUDA GPU detected - installing PyTorch with CUDA 12.1...
    pip install -q torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 ^
        --index-url https://download.pytorch.org/whl/cu121
) else (
    echo [*] No GPU detected - installing PyTorch CPU...
    pip install -q torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 ^
        --index-url https://download.pytorch.org/whl/cpu
)
if %errorlevel% neq 0 (
    echo [X] Failed to install PyTorch.
    exit /b 1
)
echo [OK] PyTorch installed.
echo.

REM -- RL packages
echo [*] Installing RL packages...
pip install -q ^
    "gymnasium==1.2.3" ^
    "stable-baselines3==2.7.1" ^
    "tensorboard" ^
    "cloudpickle" ^
    "dill"
if %errorlevel% neq 0 ( echo [X] Failed. && exit /b 1 )
echo [OK] RL packages installed.
echo.

REM -- Evolutionary optimization
echo [*] Installing evolutionary optimization...
pip install -q ^
    "evotorch==0.6.1" ^
    "cma" ^
    "ray[default]==2.54.1"
if %errorlevel% neq 0 ( echo [X] Failed. && exit /b 1 )
echo [OK] Evolutionary packages installed.
echo.

REM -- Utilities
echo [*] Installing utilities...
pip install -q "rich" "pydot" "pandas" "tqdm"
if %errorlevel% neq 0 ( echo [X] Failed. && exit /b 1 )
echo [OK] Utilities installed.
echo.

REM -- Final verification
echo === Verification ===
echo.

python -c "mods = {'double_pendulum': 'double_pendulum', 'torch': 'torch', 'numpy': 'numpy', 'scipy': 'scipy', 'matplotlib': 'matplotlib', 'gymnasium': 'gymnasium', 'stable_baselines3': 'stable_baselines3', 'ray': 'ray', 'evotorch': 'evotorch', 'cma': 'cma', 'cv2': 'opencv-python', 'rich': 'rich', 'tqdm': 'tqdm'}; [print(f'  [OK] {v}') or __import__(k) for k,v in mods.items()]"
if %errorlevel% neq 0 (
    echo [!] Some packages may be missing - check output above.
) else (
    echo.
    echo [OK] All packages verified.
)

echo.
echo ============================================================
echo    Setup complete!
echo.
echo    Next steps:
echo      python evaluate.py
echo      python baseline.py
echo ============================================================
echo.
pause