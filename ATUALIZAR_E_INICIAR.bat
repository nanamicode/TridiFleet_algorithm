@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 goto python
py -3 scripts\desktop_start.py --update
if errorlevel 1 goto erro
exit /b 0
:python
echo Instale Python 3.11 ou superior e execute novamente.
:erro
pause
exit /b 1
