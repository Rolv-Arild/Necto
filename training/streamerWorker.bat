@echo off
REM Using sets inside of ifs annoys windows, this Setlocal fixes that
Setlocal EnableDelayedExpansion

python -m venv !LocalAppData!\necto\venv

CALL  !LocalAppData!\necto\venv\Scripts\activate.bat

python -m pip install -U git+https://github.com/Rolv-Arild/rocket-learn.git
python -m pip install -U -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

if !errorlevel! neq 0 pause & exit /b !errorlevel!

echo.
echo Do you need to pull the latest Necto change?
echo.
pause

set /p ip=Enter IP address: 
set /p password=Enter password: 

echo.
echo ###################################
echo ### Launching Streamer Session! ###
echo ###################################
echo.

REM python worker.py NectoStreamer !ip! !password! False stream
python worker.py NectoStreamer !ip! !password! --streamer_mode --force_match_size 2

REM How to run stream and specify match size, that last arg here vv
REM python worker.py NectoStreamer !ip! !password! --streamer_mode --force_match_size 2

pause
