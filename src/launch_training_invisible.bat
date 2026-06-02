@echo off
REM ============================================================================
REM launch_training_invisible.bat
REM
REM Launches the 48-run training in the background:
REM   - HIGH process priority + all 16 CPU cores
REM   - No GPU memory growth flag -> TF grabs full 24 GB VRAM, others get OOM
REM   - Watchdog auto-restarts the python process if it dies
REM   - All output goes to OneDrive log so you can monitor from home
REM
REM Run by:  double-click  OR  type the path in cmd  OR  Task Scheduler
REM ============================================================================
set "REPO=C:\Users\PA Lab\Documents\wildfire-bc-bilstm-pso"

REM Detached, hidden, high priority, all CPU cores
start "" /B /HIGH /AFFINITY FFFF "%REPO%\src\watchdog_launcher.bat"

echo Training launched in background.
echo Log: %OneDrive%\wildfire_training.log
echo You can close this window -- training keeps running.
timeout /T 5 >nul
