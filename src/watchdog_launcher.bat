@echo off
REM ============================================================================
REM watchdog_launcher.bat
REM
REM Self-reinvokes with output redirected to a log file in OneDrive, then
REM loops the headless training script. Restarts within 30 seconds of any
REM crash. Resume is automatic -- skips combos with existing metrics_summary.
REM Exits cleanly when all 48 metrics_summary.json files exist.
REM ============================================================================

REM Self-redirect trick: first call lands here, re-invokes self with redirect
if "%1"=="_RUN" goto :MAIN
"%~f0" _RUN >> "%OneDrive%\wildfire_training.log" 2>&1
exit /B

:MAIN
set "REPO=C:\Users\PA Lab\Documents\wildfire-bc-bilstm-pso"
set "PY=C:\Users\PA Lab\miniconda3\envs\wildfire\python.exe"
set "SCRIPT=%REPO%\src\revision_step3_HEADLESS.py"
set "RESULTS_DIR=%REPO%\revision_c8c11\05_Model_Results"

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

REM Count completed combos
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
