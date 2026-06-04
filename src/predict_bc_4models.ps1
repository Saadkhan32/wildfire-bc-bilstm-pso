# ============================================================
#  predict_bc_4models.ps1   (v2 - desktop + laptop compatible)
#  Apply 4 winning models to BC raster stack -> 4 susceptibility GeoTIFFs.
#  - Auto-detects wildfire env on desktop (anaconda3) or laptop (ESRI conda)
#  - Forces CPU on AMD/DirectML desktop (LSTM/BiLSTM CudnnRNN incompatible)
# ============================================================
$ErrorActionPreference = "Continue"
$REPO = "$env:USERPROFILE\Documents\wildfire-bc-bilstm-pso"

Write-Host ""
Write-Host "=== Locate wildfire conda env ===" -ForegroundColor Cyan
$candidates = @(
    "$env:USERPROFILE\AppData\Local\ESRI\conda\envs\wildfire",   # laptop
    "$env:USERPROFILE\anaconda3\envs\wildfire",                  # desktop
    "$env:USERPROFILE\miniconda3\envs\wildfire"
)
$CONDA_ENV = $null
foreach ($c in $candidates) {
    if (Test-Path "$c\python.exe") {
        $CONDA_ENV = $c
        Write-Host "  Found: $CONDA_ENV" -ForegroundColor Green
        break
    }
}
if (-not $CONDA_ENV) {
    Write-Host "  FAIL: no wildfire env found in standard locations." -ForegroundColor Red
    exit 1
}
$PY = "$CONDA_ENV\python.exe"

# CUDA DLL PATH (harmless if env has no CUDA — just makes laptop GPU work)
$env:PATH = "$CONDA_ENV;$CONDA_ENV\Library\mingw-w64\bin;$CONDA_ENV\Library\usr\bin;$CONDA_ENV\Library\bin;$CONDA_ENV\Scripts;$CONDA_ENV\bin;$env:PATH"
$env:CONDA_PREFIX = $CONDA_ENV
$env:CONDA_DEFAULT_ENV = "wildfire"

# Detect if this env has DirectML (AMD GPU); if so, force CPU
$dmlCheck = & $PY -c "import os, glob; libdir = os.path.join(os.environ['CONDA_PREFIX'], 'Lib', 'site-packages', 'tensorflow-plugins'); has_dml = any('directml' in f.lower() for f in glob.glob(os.path.join(libdir, '*'))) if os.path.isdir(libdir) else False; print('DML' if has_dml else 'NO_DML')" 2>$null
if ($dmlCheck -match "DML") {
    Write-Host "  DirectML pluggable device detected -> forcing CPU mode (LSTM/BiLSTM CudnnRNN unsupported on AMD)" -ForegroundColor Yellow
    $env:FORCE_CPU = "1"
} else {
    Write-Host "  No DirectML detected; using GPU if available" -ForegroundColor Green
    Remove-Item Env:\FORCE_CPU -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "=== Check TF and rasterio availability ===" -ForegroundColor Cyan
& $PY -c "import tensorflow as tf; print('  TF', tf.__version__)" 2>&1 | Out-String -Stream | Select-String "TF" | ForEach-Object { Write-Host $_ -ForegroundColor Green }
$rioCheck = & $PY -c "import rasterio; print('rasterio', rasterio.__version__)" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  rasterio missing - installing..." -ForegroundColor Yellow
    & $PY -m pip install rasterio --quiet
    & $PY -c "import rasterio; print('  rasterio', rasterio.__version__)"
} else {
    Write-Host "  $rioCheck" -ForegroundColor Green
}

Push-Location $REPO
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Generate 4 BC susceptibility rasters" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
& $PY "src\predict_bc_4models.py"
$rc = $LASTEXITCODE
Pop-Location

if ($rc -ne 0) {
    Write-Host ""
    Write-Host "FAILED with exit code $rc" -ForegroundColor Red
    exit $rc
}

Write-Host ""
Write-Host "Saved to revision_c8c11\02_GIS_Output\:" -ForegroundColor Green
Get-ChildItem "$REPO\revision_c8c11\02_GIS_Output\BC_susceptibility_*.tif" 2>$null |
    Select-Object Name, @{n='SizeMB';e={[math]::Round($_.Length/1MB,1)}} | Format-Table -AutoSize
