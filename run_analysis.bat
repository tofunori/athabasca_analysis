@echo off
REM Multi-glacier analysis script for Windows
REM Usage: run_analysis.bat [glacier_name]

setlocal enabledelayedexpansion

REM Default glacier if none specified
if "%1"=="" (
    set GLACIER=Athabasca
) else (
    set GLACIER=%1
)

REM Check if config.yaml exists
if not exist "config.yaml" (
    echo Error: config.yaml not found!
    exit /b 1
)

echo Running analysis for %GLACIER% glacier...

REM Create output directories
mkdir executed 2>nul
mkdir results 2>nul
mkdir figs 2>nul
mkdir reports 2>nul

echo Executing notebook with Papermill...

REM Run analysis with Papermill
papermill compare_albedo.ipynb "executed/%GLACIER%_analysis.ipynb" ^
    -p GLACIER "%GLACIER%" ^
    -p CSV_PATH "data/csv/iceAWS_Atha_albedo_daily_20152020_filled_clean.csv" ^
    -p MASK_PATH "data/masks/Athabasca.tif" ^
    -p LAT 52.19 ^
    -p LON -117.28 ^
    -p OUTLIER_THRESHOLD 2.5 ^
    -p TARGET_MONTHS "[6, 7, 8, 9]"

if %errorlevel% neq 0 (
    echo Error: Papermill execution failed!
    exit /b 1
)

echo Converting to HTML report...

REM Convert to HTML report
jupyter nbconvert --to html "executed/%GLACIER%_analysis.ipynb" ^
    --output-dir reports ^
    --output "%GLACIER%_report.html"

if %errorlevel% neq 0 (
    echo Error: HTML conversion failed!
    exit /b 1
)

echo Analysis complete!
echo Results saved in:
echo   - Executed notebook: executed/%GLACIER%_analysis.ipynb
echo   - HTML report: reports/%GLACIER%_report.html
echo   - Figures: figs/%GLACIER%/
echo   - Results: results/%GLACIER%/

pause
