@echo off
REM Windows 环境变量设置脚本
REM 用于设置 OpenXLab 凭证

echo =====================================================
echo OpenXLab 凭证设置
echo =====================================================
echo.
echo 此脚本将临时设置 OpenXLab 凭证环境变量
echo （仅在当前命令行会话有效）
echo.
echo 如需永久保存，请选择永久设置选项
echo.
echo =====================================================
echo.

set AK=your_access_key
set SK=your_key

echo [选项 1] 临时设置（当前会话）
echo [选项 2] 永久设置（写入用户环境变量）
echo [选项 3] 显示当前凭证
echo [选项 4] 退出
echo.

set /p choice=请选择操作 (1/2/3/4):

if "%choice%"=="1" goto TEMP
if "%choice%"=="2" goto PERMANENT
if "%choice%"=="3" goto SHOW
if "%choice%"=="4" goto END

:TEMP
echo.
echo 设置临时环境变量...
set OPENXLAB_AK=%AK%
set OPENXLAB_SK=%SK%
echo.
echo ✓ 环境变量已设置（当前会话有效）
echo.
echo AK: %OPENXLAB_AK%
echo SK: %OPENXLAB_SK%
echo.
echo 提示: 关闭此命令行窗口后环境变量将失效
echo 现在可以运行下载脚本了:
echo   python download_ccpd_windows.py
echo.
goto END

:PERMANENT
echo.
echo 警告: 永久设置将修改系统环境变量
echo.
set /p confirm=确认永久设置? (y/n):

if /i not "%confirm%"=="y" (
    echo 已取消
    goto END
)

echo.
echo 正在设置永久环境变量...
setx OPENXLAB_AK "%AK%"
setx OPENXLAB_SK "%SK%"

if %errorlevel% equ 0 (
    echo.
    echo ✓ 环境变量已永久保存
    echo.
    echo AK: %AK%
    echo SK: %SK%
    echo.
    echo 提示: 需要重新打开命令行窗口才能生效
) else (
    echo.
    echo ✗ 设置失败，请以管理员身份运行此脚本
)
echo.
goto END

:SHOW
echo.
echo 当前环境变量:
echo.
echo OPENXLAB_AK: %OPENXLAB_AK%
echo OPENXLAB_SK: %OPENXLAB_SK%
echo.
if "%OPENXLAB_AK%"=="" (
    echo ⚠ 环境变量未设置
)
goto END

:END
echo =====================================================
pause
