@echo off
REM ============================================================================
REM watchdog_launcher.bat
REM
REM Self-redirects output to OneDrive log, then loops the headless training.
REM Sets up wildfire conda env PATH so CUDA DLLs are found (cudart64_110.dll,
REM cudnn64_8.dll, etc. live in <env>\Library\bin and TF needs them on PATH).
REM ============================================================================

REM Self-redirect: first call re-invokes self with stdout/stderr -> OneDrive log
if "%1"=="_RUN" goto :MAIN
"%~f0" _RUN >> "%OneDrive%\wildfire_training.log" 2>&1
exit /B

:MAIN
set "REPO=C:\Users\PA Lab\Documents\wildfire-bc-bilstm-pso"
set "CONDA_ENV=C:\Users\PA Lab\miniconda3\envs\wildfire"
set "PY=%CONDA_ENV%\python.exe"
set "SCRIPT=%REPO%\src\revision_step3_HEADLESS.py"
set "RESULTS_DIR=%REPO%\revision_c8c11\05_Model_Results"

REM Prepend conda env paths so CUDA DLLs (cudart64_110, cudnn64_8, etc.) load.
REM Library\bin is where cudatoolkit + cudnn .dll files live.
set "PATH=%CONDA_ENV%;%CONDA_ENV%\Library\mingw-w64\bin;%CONDA_ENV%\Library\usr\bin;%CONDA_ENV%\Library\bin;%CONDA_ENV%\Scripts;%CONDA_ENV%\bin;%PATH%"
set "CONDA_PREFIX=%CONDA_ENV%"
set "CONDA_DEFAULT_ENV=wildfire"

set /A ATTEMPT=0
set /A MAX_ATTEMPTS=99

cd /d "%REPO%"

:LOOP
set /A ATTEMPT+=1
echo.
echo ============================================================================
echo WATCHDOG ATTEMPT #%ATTEMPT%   %DATE% %TIME%
echo ============================================================================

"%PY%" "%SCRIPT%"
set "RC=%ERRORLEVEL%"

echo.
echo Python exited with code %RC% at %DATE% %TIME%

set /A DONE=0
if exist "%RESULTS_DIR%" (
    for /R "%RESULTS_DIR%" %%f in (metrics_summary.json) do set /A DONE+=1
)
echo Completed combos: %DONE% / 48

if %DONE% GEQ 48 (
    echo ============================================================================
    echo ALL 48 RUNS COMPLETE -- watchdog exiting cleanly.
    echo ============================================================================
    exit /B 0
)

if %ATTEMPT% GEQ %MAX_ATTEMPTS% (
    echo ============================================================================
    echo MAX ATTEMPTS REACHED -- watchdog giving up.
    echo ============================================================================
    exit /B 1
)

echo Restarting in 30 seconds (resume from last completed combo)...
timeout /T 30 >nul
goto LOOP
