# ============================================================
#  build_figures_roc_shap.ps1
#  One-shot: ROC train/test plot + SHAP beeswarm
#  Uses the wildfire conda env (ESRI path); auto-sets CUDA DLL PATH.
# ============================================================
$ErrorActionPreference = "Continue"
$REPO = "$env:USERPROFILE\Documents\wildfire-bc-bilstm-pso"
$CONDA_ENV = "$env:USERPROFILE\AppData\Local\ESRI\conda\envs\wildfire"
$PY  = "$CONDA_ENV\python.exe"

if (-not (Test-Path $PY)) {
    Write-Host "FAIL: python.exe not found at $PY" -ForegroundColor Red
    exit 1
}

# CUDA DLLs on PATH
$env:PATH = "$CONDA_ENV;$CONDA_ENV\Library\mingw-w64\bin;$CONDA_ENV\Library\usr\bin;$CONDA_ENV\Library\bin;$CONDA_ENV\Scripts;$CONDA_ENV\bin;$env:PATH"
$env:CONDA_PREFIX = $CONDA_ENV
$env:CONDA_DEFAULT_ENV = "wildfire"

Write-Host ""
Write-Host "=== Verify TF + SHAP + helper deps ===" -ForegroundColor Cyan
& $PY -c "import tensorflow as tf; print('TF', tf.__version__)"

# shap + its mandatory deps. --no-deps stops pip from clobbering scipy/sklearn etc.
$shapCheck = & $PY -c "import shap; print('SHAP', shap.__version__)" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  shap import failed. Installing shap + lightweight deps..." -ForegroundColor Yellow
    & $PY -m pip install shap slicer cloudpickle numba tqdm packaging --no-deps --quiet
    $shapCheck = & $PY -c "import shap; print('SHAP', shap.__version__)" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  shap STILL fails after install:" -ForegroundColor Red
        Write-Host "$shapCheck" -ForegroundColor Red
        exit 2
    }
}
Write-Host "  $shapCheck" -ForegroundColor Green

Push-Location $REPO

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " STEP 1 -- ROC train/test plot (4 models)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
& $PY "src\build_roc_train_test.py"
$rocRc = $LASTEXITCODE
if ($rocRc -ne 0) {
    Write-Host ""
    Write-Host "STEP 1 FAILED with exit code $rocRc. Aborting." -ForegroundColor Red
    Pop-Location
    exit $rocRc
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " STEP 2 -- SHAP beeswarm (BiLSTM-PSO winning model)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
& $PY "src\build_shap_beeswarm.py"
$shapRc = $LASTEXITCODE
if ($shapRc -ne 0) {
    Write-Host ""
    Write-Host "STEP 2 FAILED with exit code $shapRc." -ForegroundColor Red
    Pop-Location
    exit $shapRc
}

Pop-Location

Write-Host ""
Write-Host "All figures saved to revision_c8c11\03_Figures\:" -ForegroundColor Green
Write-Host "  Fig_ROC_4models_train_test.pdf / .png"
Write-Host "  Fig_SHAP_BiLSTM_PSO_beeswarm.pdf / .png"
Write-Host ""
