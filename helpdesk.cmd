@echo off
setlocal
powershell.exe -NoProfile -InputFormat None -ExecutionPolicy Bypass -File "%~dp0helpdesk.ps1" %*
exit /b %errorlevel%
