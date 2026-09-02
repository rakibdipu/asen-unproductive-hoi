@echo off
title আসেন Unproductive হই (Asen Unproductive Hoi)
color 0C
echo =======================================================================
echo   🛋️  আসেন Unproductive হই (Asen Unproductive Hoi)
echo   The Ultimate AI Procrastination & Distraction Engine
echo =======================================================================
echo.
echo [1/2] Launching FastAPI Backend on http://localhost:8000 ...
start "Asen-Backend" cmd /k "python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"

echo [2/2] Launching React 19 Frontend on http://localhost:5173 ...
cd frontend
start "Asen-Frontend" cmd /k "npm run dev"

echo.
echo =======================================================================
echo   ✅ Both servers are starting up!
echo   👉 Open your browser at: http://localhost:5173
echo   👉 API Documentation at:  http://localhost:8000/docs
echo =======================================================================
timeout /t 5
start http://localhost:5173
pause
