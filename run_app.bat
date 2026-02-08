@echo off
echo Installing dependencies using Python 3.13...
py -3.13 -m pip install -r requirements.txt

echo Starting the application with Python 3.13...
py -3.13 app.py
pause
