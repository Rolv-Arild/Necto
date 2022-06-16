@echo off
REM Using sets inside of ifs annoys windows, this Setlocal fixes that
Setlocal EnableDelayedExpansion

REM optional argument to launch multiple workers at once
set instance_num=1
set tending=0

REM go through arguments
for %%a in (%*) do (
    REM check for tending flag
    if "%%a"=="\t" set tending=1

    REM check for instance number (positive integers)
    if 1%%a EQU +1%%a set instance_num=%%a
)


WHERE git
if !errorlevel! neq 0 echo git required, download here: https://git-scm.com/download/win
if !errorlevel! neq 0 pause & exit

if exist !APPDATA!\bakkesmod\ (
    echo Bakkesmod located at !APPDATA!\bakkesmod\ 
    goto :done
    
) else (
    echo \nBakkesmod not found at !APPDATA!\bakkesmod\ 
    echo * If you've already installed it elsewhere, you're fine *
    set /p choice=Download Bakkesmod[y/n]?:
    
    if /I "!choice!" EQU "Y" goto :install
    if /I "!choice!" EQU "N" goto :no_install
)

    :install
echo Downloading Bakkesmod
curl.exe -L --output !USERPROFILE!\Downloads\BakkesModSetup.zip --url https://github.com/bakkesmodorg/BakkesModInjectorCpp/releases/latest/download/BakkesModSetup.zip
tar -xf !USERPROFILE!\Downloads\BakkesModSetup.zip -C !USERPROFILE!\Downloads\
!USERPROFILE!\Downloads\BakkesModSetup.exe

if !errorlevel! neq 0 echo \n*** Problem with Bakkesmod installation. Manually install and try again ***\n
if !errorlevel! neq 0 pause & exit /b !errorlevel!

echo Bakkesmod installed!
goto :done

:no_install
goto :done
    
    :done

python -m venv !LocalAppData!\necto\venv

CALL !LocalAppData!\necto\venv\Scripts\activate.bat

REM python -m pip install --upgrade pip

python -m pip install -U git+https://github.com/Rolv-Arild/rocket-learn.git
python -m pip install -U -r requirements.txt -f https://download.pytorch.org/whl/cu113

if !errorlevel! neq 0 pause & exit /b !errorlevel!

REM Automatically pull latest version, avoid stashing if no changes to avoid errors
for /f %%i in ('call git status --porcelain --untracked-files=no') do set stash=%%i
if not [%stash%] == [] (
    git stash
    git checkout master
    git pull origin master
    git stash apply
) else (
    git checkout master
    git pull origin master
)

set /p helper_name=Enter name:
set /p ip=Enter IP address:
set /p password=Enter password:

echo.
echo #########################
echo ### Launching Worker! ###
echo #########################
echo.


    :process_launch
set process_list=
for /L %%i in (1, 1, !instance_num!) do (
    set title=NectoWorker_%%i
    set process_list=!process_list!;!title!

    REM launch workers in new cmd to make tracking errors easier
    start "!title!" cmd /c python worker.py !helper_name! !ip! !password! --compress ^& pause
    timeout /t 45 /nobreak >nul
)

REM if we tend, wait, then close and do it all over again
if !tending! EQU 1 (
    REM 12 hours in seconds
    timeout /t 43200 /nobreak >nul

    echo Closing worker instances
    REM kill processes and relaunch
    for %%i in (!process_list!) do (
        echo %%i
        taskkill /FI "WindowTitle eq %%i" /T /F
        timeout /t 1 >nul
    )

    REM double check that all rocket league instances are killed
    taskkill /FI "WindowTitle eq Rocket*" /T /F

    echo Closed. Waiting 5 minutes to ensure proper teardown.
    timeout /t 300 /nobreak >nul
    echo Relaunching worker instances.
    goto :process_launch
)

