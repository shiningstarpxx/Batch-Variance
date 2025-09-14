#!/bin/bash

# LLM推理非确定性复现项目安装脚本

echo "=========================================="
echo "LLM推理非确定性复现项目安装脚本"
echo "=========================================="

# 检查Python版本
echo "检查Python版本..."
python3 --version

# 创建虚拟环境（可选）
read -p "是否创建虚拟环境？(y/n): " create_venv
if [ "$create_venv" = "y" ] || [ "$create_venv" = "Y" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
    source venv/bin/activate
    echo "虚拟环境已激活"
fi

# 安装依赖
echo "安装Python依赖..."
pip install -r requirements.txt

# 创建必要的目录
echo "创建项目目录..."
mkdir -p experiments/results
mkdir -p experiments/plots

# 设置执行权限
chmod +x run_demo.py

echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "使用方法："
echo "1. 运行所有演示："
echo "   python run_demo.py --demo all"
echo ""
echo "2. 运行特定演示："
echo "   python run_demo.py --demo floating_point"
echo "   python run_demo.py --demo attention"
echo "   python run_demo.py --demo batch_invariant"
echo "   python run_demo.py --demo visualization"
echo ""
echo "3. 运行Jupyter notebooks："
echo "   jupyter notebook notebooks/"
echo ""
echo "4. 如果创建了虚拟环境，记得激活："
echo "   source venv/bin/activate"
echo "=========================================="
