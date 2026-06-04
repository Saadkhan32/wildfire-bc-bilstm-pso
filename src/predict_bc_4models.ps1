# ============================================================
#  predict_bc_4models.ps1
#  Apply 4 winning models to BC raster stack -> 4 susceptibility GeoTIFFs.
# ============================================================
$ErrorActionPreference = "Continue"
$REPO = "$env:USERPROFILE\Documents\wildfire-bc-bilstm-pso"
$CONDA_ENV = "$env:USERPROFILE\AppData\Local\ESRI\conda\envs\wildfire"
$PY  = "$CONDA_ENV\python.exe"

if (-not (Test-Path $PY)) {
    Write-Host "FAIL: python.exe not found at $PY" -ForegroundColor Red
    exit 1
}

$env:PATH = "$CONDA_ENV;$CONDA_ENV\Library\mingw-w64\bin;$CONDA_ENV\Library\usr\bin;$CONDA_ENV\Library\bin;$CONDA_ENV\Scripts;$CONDA_ENV\bin;$env:PATH"
$env:CONDA_PREFIX = $CONDA_ENV
$env:CONDA_DEFAULT_ENV = "wildfire"

Write-Host "=== Check rasterio availability ===" -ForegroundColor Cyan
$rioCheck = & $PY -c "import rasterio; print('rasterio', rasterio.__version__)" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  rasterio missing - installing..." -ForegroundColor Yellow
    & $PY -m pip install rasterio --quiet
    & $PY -c "import rasterio; print('rasterio', rasterio.__version__)"
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
