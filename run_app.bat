@echo off
title AI Churn Copilot
cd /d "%~dp0"
call .venv\Scripts\activate 2>nul
start /min cmd /c "streamlit run src/streamlit_app.py --server.headless true"
timeout /t 3 /nobreak >nul
start http://localhost:8501