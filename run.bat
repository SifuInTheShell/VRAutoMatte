@echo off
rem VRAutoMatte launcher.
rem --all-extras keeps sam2/matanyone2 installed — a plain
rem "uv run" syncs only default deps and would remove them.
cd /d "%~dp0"
uv run --all-extras vrautomatte %*
if errorlevel 1 (
    echo.
    echo VRAutoMatte exited with an error.
    pause
)
