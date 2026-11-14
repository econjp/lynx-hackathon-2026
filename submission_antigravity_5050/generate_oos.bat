@echo off
setlocal
echo ================================================================================
echo   TEAM NEWNORDICS: 50/50 MULTI-STRATEGY OOS SUBMISSION GENERATOR
echo ================================================================================

cd /d "%~dp0"

if "%~1"=="" (
    echo No parameters passed. Running baseline generation on in-sample data...
    python run_5050.py
) else if "%~2"=="" (
    echo Running with Out-of-Sample prices: %~1
    python run_5050.py --prices "..\data\close_prices_insample.csv" "%~1" --macro "..\data\macro_data_insample.csv" --out "positions.csv"
) else (
    echo Running with Out-of-Sample prices: %~1 and macro: %~2
    python run_5050.py --prices "..\data\close_prices_insample.csv" "%~1" --macro "..\data\macro_data_insample.csv" "%~2" --out "positions.csv"
)

echo.
echo ================================================================================
echo Done! Output file: %~dp0positions.csv
echo ================================================================================
pause
