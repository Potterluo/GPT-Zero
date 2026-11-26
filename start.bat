@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul
REM ============================================================================
REM GPT-Zero v4 启动脚本（Windows）
REM   优先用项目 .venv 的解释器；没有则回退系统 python。
REM   端口默认 8090（本机 8080 被 piforged 常驻占用）；可自行覆盖：
REM     start.bat --port 9090
REM     set GPTZERO_PORT=9090 ^&^& start.bat
REM   支持全部 gptzero CLI 参数（--model / --ttft-ms / --log-file ...）。
REM ============================================================================
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PY=venv\Scripts\python.exe"
    echo [提示] 未找到 .venv，使用 venv\  ^(建议运行 install.bat 统一为 .venv^)
) else (
    echo [提示] 未找到虚拟环境，使用系统 python ^(建议先运行 install.bat^)
    set "PY=python"
)

REM 端口默认 8090，除非调用方已显式指定
if not defined GPTZERO_PORT set "GPTZERO_PORT=8090"

echo ====================================
echo GPT-Zero v4 启动
echo   解释器 : !PY!
echo   端口   : !GPTZERO_PORT!   ^(http://127.0.0.1:!GPTZERO_PORT!^)
echo   参数   : %*
echo ====================================
echo.

"!PY!" server.py %*
if errorlevel 1 (
    echo.
    echo [错误] 服务异常退出 ^(退出码 !errorlevel!^)。
    echo 排查: 端口是否被占用? netstat -ano ^| findstr :!GPTZERO_PORT!
    pause
)
endlocal
