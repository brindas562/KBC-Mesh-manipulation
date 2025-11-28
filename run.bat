@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Starting cloth simulation...
python cloth_simulation.py

pause
