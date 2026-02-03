@echo off
REM CCPD 数据集快速处理脚本
REM 用于 Windows 系统一键下载、转换 CCPD 数据集

echo ======================================================
echo OpenTrafficFlow - CCPD 数据集快速处理
echo ======================================================
echo.
echo 本脚本将自动执行以下步骤：
echo 1. 下载 CCPD2020 数据集 (865MB)
echo 2. 解压数据集
echo 3. 转换为 YOLO 格式
echo.
echo ======================================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Python，请先安装 Python
    pause
    exit /b 1
)

echo [1/3] 下载 CCPD2020 数据集...
echo 提示: 请准备 OpenXLab 的 AK/SK
echo.
set DOWNLOAD_CHOICE=n
set /p DOWNLOAD_CHOICE=是否现在下载数据集? (y/n):

if /i "%DOWNLOAD_CHOICE%"=="y" (
    echo.
    echo 正在启动下载脚本...
    python download_ccpd_windows.py
    echo.
) else (
    echo 跳过下载步骤
    echo.
)

echo [2/3] 转换为 YOLO 格式...
echo.

REM 检查数据集是否存在
if exist "CCPD_Datasets\CCPD2020" (
    echo 找到 CCPD2020 数据集
    echo.
    set CONVERT_CHOICE=n
    set /p CONVERT_CHOICE=是否转换为 YOLO 格式? (y/n):

    if /i "%CONVERT_CHOICE%"=="y" (
        echo.
        echo 正在转换数据集...
        echo.
        python convert_ccpd_to_yolo.py --source ./CCPD_Datasets/CCPD2020 --target ./YOLO_Data --copy
        echo.
    ) else (
        echo 跳过转换步骤
    )
) else (
    echo [提示] 未找到 CCPD2020 数据集，跳过转换步骤
    echo 请先运行下载脚本或手动下载数据集
    echo.
)

echo ======================================================
echo 处理完成!
echo ======================================================
echo.
echo 下一步:
echo 1. 检查 YOLO_Data 文件夹
echo 2. 使用 YOLOv5 训练模型
echo.
echo 示例训练命令:
echo   python train.py --data YOLO_Data/data.yaml --weights yolov5s.pt
echo.
echo ======================================================
pause
