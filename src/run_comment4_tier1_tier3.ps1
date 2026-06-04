# ============================================================
#  run_comment4_tier1_tier3.ps1
#  Run the Tier 1 + Tier 3 cross-border consistency analysis.
#  Auto-detects wildfire env on desktop / laptop and installs
#  any missing geopandas / shapely / scipy / scikit-learn deps.
# ============================================================
$ErrorActionPreference = "Continue"
$REPO = "$env:USERPROFILE\Documents\wildfire-bc-bilstm-pso"

Write-Host ""
Write-Host "=== Locate wildfire conda env ===" -ForegroundColor Cyan
$candidates = @(
    "$env:USERPROFILE\AppData\Local\ESRI\conda\envs\wildfire",
    "$env:USERPROFILE\anaconda3\envs\wildfire",
    "$env:USERPROFILE\miniconda3\envs\wildfire"
)
$CONDA_ENV = $null
foreach ($c in $candidates) {
    if (Test-Path "$c\python.exe") { $CONDA_ENV = $c; break }
}
if (-not $CONDA_ENV) { Write-Host "FAIL: no wildfire env." -ForegroundColor Red; exit 1 }
Write-Host "  Using: $CONDA_ENV" -ForegroundColor Green
$PY = "$CONDA_ENV\python.exe"

# CUDA DLL PATH (harmless for this script; consistent with other launchers)
$env:PATH = "$CONDA_ENV;$CONDA_ENV\Library\bin;$CONDA_ENV\Scripts;$env:PATH"
$env:CONDA_PREFIX = $CONDA_ENV
$env:CONDA_DEFAULT_ENV = "wildfire"

Write-Host ""
Write-Host "=== Check Python deps ===" -ForegroundColor Cyan
$deps = @{
    "rasterio"     = "rasterio"
    "geopandas"    = "geopandas"
    "shapely"      = "shapely"
    "scipy"        = "scipy"
    "scikit-learn" = "sklearn"
    "matplotlib"   = "matplotlib"
    "pandas"       = "pandas"
    "tqdm"         = "tqdm"
}
foreach ($pkg in $deps.Keys) {
    $mod = $deps[$pkg]
    $check = & $PY -c "import $mod; print('$mod OK')" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  $pkg missing -> installing..." -ForegroundColor Yellow
        & $PY -m pip install $pkg --quiet
        & $PY -c "import $mod; print('  $mod', $mod.__version__ if hasattr($mod, '__version__') else 'OK')"
    } else {
        Write-Host "  $check" -ForegroundColor Green
    }
}

Push-Location $REPO
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Run Tier 1 + Tier 3 cross-border analysis" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
& $PY "src\comment4_xborder_tier1_tier3.py"
$rc = $LASTEXITCODE
Pop-Location

if ($rc -ne 0) {
    Write-Host ""
    Write-Host "FAILED with exit code $rc" -ForegroundColor Red
    exit $rc
}

Write-Host ""
Write-Host "Outputs:" -ForegroundColor Green
Write-Host "  Fig_S6b_xborder_consistency.{pdf,png}  in revision_c8c11\03_Figures\"
Write-Host "  T_C4_xborder_tier1_tier3.json          in revision_c8c11\06_Final_Tables\"
Write-Host "  T_C4_hex_tile_values.csv               in revision_c8c11\06_Final_Tables\"
