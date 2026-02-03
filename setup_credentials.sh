#!/bin/bash
# Linux/Mac 环境变量设置脚本
# 用于设置 OpenXLab 凭证

AK="your_access_key"
SK="your_key"

echo "====================================================="
echo "OpenXLab 凭证设置"
echo "====================================================="
echo ""
echo "此脚本将设置 OpenXLab 凭证环境变量"
echo ""

# 临时设置
echo "[选项 1] 临时设置（当前会话）"
echo "[选项 2] 永久设置（写入 ~/.bashrc）"
echo "[选项 3] 永久设置（写入 ~/.zshrc）"
echo "[选项 4] 显示当前凭证"
echo "[选项 5] 退出"
echo ""

read -p "请选择操作 (1/2/3/4/5): " choice

case $choice in
    1)
        echo ""
        echo "设置临时环境变量..."
        export OPENXLAB_AK=$AK
        export OPENXLAB_SK=$SK
        echo ""
        echo "✓ 环境变量已设置（当前会话有效）"
        echo ""
        echo "AK: $OPENXLAB_AK"
        echo "SK: $OPENXLAB_SK"
        echo ""
        echo "提示: 关闭终端后环境变量将失效"
        echo "现在可以运行下载脚本了:"
        echo "  python download_ccpd.py"
        ;;
    2)
        echo ""
        echo "写入 ~/.bashrc..."
        echo "" >> ~/.bashrc
        echo "# OpenXLab 凭证" >> ~/.bashrc
        echo "export OPENXLAB_AK=$AK" >> ~/.bashrc
        echo "export OPENXLAB_SK=$SK" >> ~/.bashrc
        echo ""
        echo "✓ 环境变量已写入 ~/.bashrc"
        echo ""
        echo "AK: $AK"
        echo "SK: $SK"
        echo ""
        echo "请执行以下命令使配置生效:"
        echo "  source ~/.bashrc"
        echo "或重新打开终端"
        ;;
    3)
        echo ""
        echo "写入 ~/.zshrc..."
        echo "" >> ~/.zshrc
        echo "# OpenXLab 凭证" >> ~/.zshrc
        echo "export OPENXLAB_AK=$AK" >> ~/.zshrc
        echo "export OPENXLAB_SK=$SK" >> ~/.zshrc
        echo ""
        echo "✓ 环境变量已写入 ~/.zshrc"
        echo ""
        echo "AK: $AK"
        echo "SK: $SK"
        echo ""
        echo "请执行以下命令使配置生效:"
        echo "  source ~/.zshrc"
        echo "或重新打开终端"
        ;;
    4)
        echo ""
        echo "当前环境变量:"
        echo ""
        echo "OPENXLAB_AK: $OPENXLAB_AK"
        echo "OPENXLAB_SK: $OPENXLAB_SK"
        echo ""
        if [ -z "$OPENXLAB_AK" ]; then
            echo "⚠ 环境变量未设置"
        fi
        ;;
    5)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

echo ""
echo "====================================================="
