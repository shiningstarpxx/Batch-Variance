#!/usr/bin/env python3
"""
LLM推理非确定性复现项目主运行脚本

这个脚本提供了运行所有演示的入口点。
"""

import sys
import os
import argparse
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from floating_point import FloatingPointDemo
from attention import AttentionNondeterminismDemo
from batch_invariant import BatchInvariantDemo
from visualization import VisualizationUtils, create_sample_data

def run_floating_point_demo():
    """运行浮点数非结合性演示"""
    print("=" * 60)
    print("运行浮点数非结合性演示")
    print("=" * 60)
    
    demo = FloatingPointDemo()
    demo.run_complete_demo()
    
    print("\n浮点数非结合性演示完成！")

def run_attention_demo():
    """运行注意力机制非确定性演示"""
    print("=" * 60)
    print("运行注意力机制非确定性演示")
    print("=" * 60)
    
    demo = AttentionNondeterminismDemo(device='cpu')
    demo.run_complete_demo()
    
    print("\n注意力机制非确定性演示完成！")

def run_batch_invariant_demo():
    """运行批处理不变性演示"""
    print("=" * 60)
    print("运行批处理不变性演示")
    print("=" * 60)
    
    demo = BatchInvariantDemo(device='cpu')
    demo.run_complete_demo()
    
    print("\n批处理不变性演示完成！")

def run_visualization_demo():
    """运行可视化演示"""
    print("=" * 60)
    print("运行可视化演示")
    print("=" * 60)
    
    viz = VisualizationUtils()
    sample_data = create_sample_data()
    
    # 创建综合仪表板
    viz.create_comprehensive_dashboard(sample_data, 'experiments/plots/comprehensive_dashboard.png')
    
    # 创建交互式图表
    viz.create_interactive_plot(sample_data, 'experiments/plots/interactive_dashboard.html')
    
    # 创建其他图表
    viz.create_heatmap(None, "示例热力图", "experiments/plots/heatmap.png")
    viz.create_correlation_matrix(None, "experiments/plots/correlation_matrix.png")
    viz.create_3d_surface(None, None, None, "示例3D表面图", "experiments/plots/3d_surface.png")
    
    print("\n可视化演示完成！")

def run_all_demos():
    """运行所有演示"""
    print("=" * 60)
    print("运行所有演示")
    print("=" * 60)
    
    # 创建必要的目录
    os.makedirs('experiments/plots', exist_ok=True)
    
    # 运行所有演示
    run_floating_point_demo()
    run_attention_demo()
    run_batch_invariant_demo()
    run_visualization_demo()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)
    print("结果已保存到 experiments/plots/ 目录")
    print("可以查看以下文件：")
    print("- floating_point_demo.png: 浮点数非结合性演示")
    print("- attention_differences.png: 注意力机制差异")
    print("- attention_performance.png: 注意力机制性能")
    print("- batch_invariance.png: 批处理不变性对比")
    print("- batch_invariant_performance.png: 批处理不变性性能")
    print("- comprehensive_dashboard.png: 综合仪表板")
    print("- interactive_dashboard.html: 交互式仪表板")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LLM推理非确定性复现项目')
    parser.add_argument('--demo', choices=['floating_point', 'attention', 'batch_invariant', 
                                         'visualization', 'all'], 
                       default='all', help='选择要运行的演示')
    
    args = parser.parse_args()
    
    # 创建必要的目录
    os.makedirs('experiments/plots', exist_ok=True)
    
    if args.demo == 'floating_point':
        run_floating_point_demo()
    elif args.demo == 'attention':
        run_attention_demo()
    elif args.demo == 'batch_invariant':
        run_batch_invariant_demo()
    elif args.demo == 'visualization':
        run_visualization_demo()
    elif args.demo == 'all':
        run_all_demos()
    else:
        print(f"未知的演示类型: {args.demo}")
        sys.exit(1)

if __name__ == "__main__":
    main()
