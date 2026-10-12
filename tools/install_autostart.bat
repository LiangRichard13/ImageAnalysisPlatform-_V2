@echo off
rem One-click autostart config (double-click; requests admin via UAC)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_autostart.ps1"
