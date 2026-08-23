@echo off
chcp 65001 >nul
title Установка W1x CLI

echo ===================================================
echo   Установка W1x CLI — Терминального ИИ для Ollama
echo ===================================================
echo.

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ОШИБКА] Python не найден! Установите Python 3.10+ с галочкой "Add Python to PATH".
    pause
    exit /b 1
)

echo [1/3] Обновление pip...
python -m pip install --upgrade pip

echo [2/3] Установка зависимостей (rich, prompt_toolkit, ollama, duckduckgo)...
python -m pip install -r "%~dp0requirements.txt"

echo [3/3] Установка пакета в систему...
python -m pip install -e "%~dp0."

echo.
echo ===================================================
echo   ✓ Установка успешно завершена!
echo   Теперь вы можете запускать чат командой: w1x
echo ===================================================
echo.
pause
