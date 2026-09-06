@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo Instale Python 3.11 ou superior e execute novamente.
  pause
  exit /b 1
)
if not exist .venv\Scripts\python.exe py -3 -m venv .venv
if errorlevel 1 goto erro
.venv\Scripts\python.exe -m pip install -e .
if errorlevel 1 goto erro
if not exist .env copy .env.example .env >nul
echo Abra http://localhost:8000 no navegador.
echo Login padrao: admin / tridifleet-local. Personalize no arquivo .env.
echo Pode fechar a aba. Mantenha esta janela aberta para continuar simulando.
.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --env-file .env
if errorlevel 1 goto erro
exit /b 0
:erro
echo Nao foi possivel iniciar. Confira a mensagem acima.
pause
exit /b 1
