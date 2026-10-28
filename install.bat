@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul
REM ============================================================================
REM GPT-Zero v3 环境安装脚本（Windows）
REM   - 优先复用已有 .venv（项目约定），否则创建
REM   - 依赖以 pyproject.toml 为准（editable 安装 + dev extras）
REM   - 走清华 PyPI 镜像，避免墙内超时
REM   - transformers 可选：装不上不影响服务（自动 heuristic 回退）
REM ============================================================================
cd /d "%~dp0"

set "MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple"
set "MIRROR_HOST=pypi.tuna.tsinghua.edu.cn"
set "VENV_DIR=.venv"

echo ====================================
echo GPT-Zero v3 环境安装
echo 目录: %CD%
echo ====================================
echo.

REM ---- [0/5] 检查 Python ----
echo [0/5] 检查 Python...
where python >nul 2>nul
if errorlevel 1 (
    echo 错误: 找不到 python，请先安装 Python 3.10+ 并加入 PATH。
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PYVER=%%v"
echo      Python !PYVER!  OK

REM ---- [1/5] 虚拟环境 ----
if exist "%VENV_DIR%\Scripts\python.exe" (
    echo [1/5] 复用已有虚拟环境 %VENV_DIR%
) else (
    echo [1/5] 创建虚拟环境 %VENV_DIR% ...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo 错误: 创建虚拟环境失败。
        pause
        exit /b 1
    )
)

set "PY=%VENV_DIR%\Scripts\python.exe"

REM ---- [2/5] 升级 pip ----
echo.
echo [2/5] 升级 pip / setuptools / wheel ...
"%PY%" -m pip install --upgrade pip setuptools wheel -i %MIRROR% --trusted-host %MIRROR_HOST%
if errorlevel 1 (
    echo 警告: 升级 pip 失败，继续尝试安装依赖...
)

REM ---- [3/5] 安装项目依赖（editable + dev）----
echo.
echo [3/5] 安装项目依赖 ^(pip install -e ".[dev]"^) ...
"%PY%" -m pip install -e ".[dev]" -i %MIRROR% --trusted-host %MIRROR_HOST%
if errorlevel 1 (
    echo 错误: 依赖安装失败。可重试: "%PY%" -m pip install -e ".[dev]"
    pause
    exit /b 1
)

REM ---- [4/5] 可选：transformers（精确 token 计数）----
echo.
echo [4/5] 安装可选依赖 transformers ^(精确 token 计数^) ...
"%PY%" -m pip install "transformers>=4.40,<6" -i %MIRROR% --trusted-host %MIRROR_HOST%
if errorlevel 1 (
    echo     警告: transformers 安装失败，服务会自动回退到 heuristic 分词。
    echo     这不影响功能，只影响 prompt_token 的精确度。
) else (
    echo     transformers 已安装，将使用本地 deepseek_v3_tokenizer/ 精确计数。
)

REM ---- [5/5] 自检 ----
echo.
echo [5/5] 自检...
"%PY%" -c "import fastapi,uvicorn,pydantic;print('    fastapi/uvicorn/pydantic OK')"
if errorlevel 1 (
    echo 错误: 依赖导入失败。
    pause
    exit /b 1
)
"%PY%" -m compileall -q gptzero server.py >nul 2>nul
if errorlevel 1 (
    echo 警告: compileall 报错，请检查源码。
) else (
    echo     import/编译检查 OK
)

echo.
echo ====================================
echo 安装完成
echo ====================================
echo.
echo 启动服务:
echo   start.bat                          ^(默认 8090 端口^)
echo   %PY% server.py --port 8090
echo.
echo 跑测试:
echo   %PY% -m pytest tests -q
echo.
echo Docker ^(纯离线镜像^):
echo   docker compose up -d mock          ^(宿主机 8090^)
echo   .\scripts\offline_image.ps1        ^(构建 + 导出 tar^)
echo.
pause
