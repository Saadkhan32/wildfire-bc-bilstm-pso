# ============================================================
#  gap_fill_8runs.ps1   (v3 -- ESRI conda env)
#  Resume the 8 missing cells of the 48-run sensitivity grid.
#  Resumable: skips 40 already-complete cells automatically.
#  Retries up to 3 times to defeat transient CUDA_ERROR_UNKNOWN.
# ============================================================
#  Run from:  C:\Users\saadz\Documents\wildfire-bc-bilstm-pso\
#  PowerShell:  .\src\gap_fill_8runs.ps1
# ============================================================

$ErrorActionPreference = "Continue"
$REPO = "$env:USERPROFILE\Documents\wildfire-bc-bilstm-pso"
$RES  = "$REPO\revision_c8c11\05_Model_Results"

# ---------- Hard-coded wildfire env (ESRI conda) ----------
$CONDA_ENV = "$env:USERPROFILE\AppData\Local\ESRI\conda\envs\wildfire"
$PY        = "$CONDA_ENV\python.exe"

Write-Host ""
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host " STEP 0 -- Verify wildfire env" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

if (-not (Test-Path $PY)) {
    Write-Host "  FAIL: python.exe not found at $PY" -ForegroundColor Red
    exit 1
}
Write-Host "  Using python.exe: $PY" -ForegroundColor Green

# Prepend conda env paths so CUDA DLLs (cudart64, cudnn) load from Library\bin
$env:PATH = "$CONDA_ENV;$CONDA_ENV\Library\mingw-w64\bin;$CONDA_ENV\Library\usr\bin;$CONDA_ENV\Library\bin;$CONDA_ENV\Scripts;$CONDA_ENV\bin;$env:PATH"
$env:CONDA_PREFIX      = $CONDA_ENV
$env:CONDA_DEFAULT_ENV = "wildfire"

# Verify TF is importable
Write-Host "  Verifying TensorFlow available..." -ForegroundColor Cyan
$tfCheck = & $PY -c "import tensorflow as tf; print('TF', tf.__version__)" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  FAIL: tensorflow not importable in this env." -ForegroundColor Red
    Write-Host "  Output: $tfCheck" -ForegroundColor Red
    exit 1
}
Write-Host "  OK: $tfCheck" -ForegroundColor Green

# Verify GPU is visible
Write-Host "  Checking GPU visibility..." -ForegroundColor Cyan
$gpuCheck = & $PY -c "import tensorflow as tf; gpus=tf.config.list_physical_devices('GPU'); print('GPUs:', gpus)" 2>&1
Write-Host "  $gpuCheck" -ForegroundColor Green

Write-Host ""
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host " STEP 1 -- Clean failed / orphan seed directories" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

$failedDirs = @(
    "$RES\LSTM_PSO\thr70\seed909",
    "$RES\LSTM_PSO\thr100\seed42",
    "$RES\LSTM_PSO\thr200\seed42",
    "$RES\BiLSTM_PSO\thr200\seed42",
    "$RES\__nonpso_logs__\thr100_seed42",
    "$RES\__nonpso_logs__\thr200_seed42",
    "$RES\LSTM\thr100\seed42",
    "$RES\LSTM\thr200\seed42",
    "$RES\BiLSTM\thr100\seed42",
    "$RES\BiLSTM\thr200\seed42"
)

foreach ($d in $failedDirs) {
    if (Test-Path $d) {
        Write-Host "  removing: $d" -ForegroundColor Yellow
        Remove-Item -Recurse -Force $d
    } else {
        Write-Host "  (already absent): $d"
    }
}

Write-Host ""
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host " STEP 2 -- Pre-flight: count current metrics" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

$before = (Get-ChildItem -Path $RES -Recurse -Filter "metrics_summary.json" -ErrorAction SilentlyContinue).Count
Write-Host "  metrics_summary.json count BEFORE gap-fill: $before  (expect 40)" -ForegroundColor Green

Write-Host ""
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host " STEP 3 -- Launch headless trainer (auto-resumable)" -ForegroundColor Cyan
Write-Host " Will retry up to 3 times to beat transient CUDA errors" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

$maxAttempts = 3
$target = 48

for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
    Write-Host ""
    Write-Host ">>>>>>>>>>>>>>>>  ATTEMPT $attempt of $maxAttempts  <<<<<<<<<<<<<<<<" -ForegroundColor Magenta
    Write-Host ""

    Push-Location $REPO
    & $PY "src\revision_step3_HEADLESS.py" --mode all --verbose --python-exe $PY
    $rc = $LASTEXITCODE
    Pop-Location

    $now = (Get-ChildItem -Path $RES -Recurse -Filter "metrics_summary.json" -ErrorAction SilentlyContinue).Count
    Write-Host ""
    Write-Host "  After attempt ${attempt}: $now of $target metrics_summary.json present" -ForegroundColor Green

    if ($now -ge $target) {
        Write-Host ""
        Write-Host "  ALL 48 RUNS COMPLETE." -ForegroundColor Green
        break
    } else {
        Write-Host ""
        Write-Host "  $(($target - $now)) cells still missing -- will retry after a 30s cooldown" -ForegroundColor Yellow
        Start-Sleep -Seconds 30
    }
}

Write-Host ""
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host " STEP 4 -- Final inventory" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

$final = Get-ChildItem -Path $RES -Recurse -Filter "metrics_summary.json" -ErrorAction SilentlyContinue
Write-Host "  Final count: $($final.Count) of $target" -ForegroundColor Green
Write-Host ""

$expectedCells = @()
$seedsThr70 = @(42, 101, 202, 303, 404, 505, 606, 707, 808, 909)
$thrAtSeed42 = @(100, 200)
$models = @("LSTM", "BiLSTM", "LSTM_PSO", "BiLSTM_PSO")

foreach ($m in $models) {
    foreach ($s in $seedsThr70) {
        $expectedCells += "$m\thr70\seed$s"
    }
    foreach ($t in $thrAtSeed42) {
        $expectedCells += "$m\thr$t\seed42"
    }
}

$present = @{}
foreach ($f in $final) {
    $rel = $f.FullName.Replace("$RES\","").Replace("\metrics_summary.json","")
    $present[$rel] = $true
}

$missing = @()
foreach ($c in $expectedCells) {
    if (-not $present.ContainsKey($c)) { $missing += $c }
}

if ($missing.Count -eq 0) {
    Write-Host "  All 48 cells present. Sync results to desktop next." -ForegroundColor Green
} else {
    Write-Host "  Still missing $($missing.Count) cells:" -ForegroundColor Red
    foreach ($c in $missing) { Write-Host "    - $c" -ForegroundColor Red }
    Write-Host ""
    Write-Host "  -> Re-run this script, or inspect the run_log.txt in those folders." -ForegroundColor Yellow
}
Write-Host ""
