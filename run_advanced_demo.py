#!/usr/bin/env python3
"""
高级演示脚本

这个脚本展示了Mac MPS计算支持和更多维度的验证功能。
"""

import argparse
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from device_manager import device_manager, print_device_info, benchmark_devices
from floating_point import FloatingPointDemo
from attention import AttentionNondeterminismDemo
from batch_invariant import BatchInvariantDemo
from advanced_analysis import AdvancedAnalysis

def run_device_benchmark():
    """运行设备基准测试"""
    print("=" * 60)
    print("设备基准测试")
    print("=" * 60)
    
    # 打印设备信息
    print_device_info()
    
    # 运行基准测试
    print("\n运行设备基准测试...")
    results = benchmark_devices(size=1000)
    
    print("\n基准测试结果:")
    for device, result in results.items():
        print(f"{device.upper()}:")
        print(f"  平均时间: {result['avg_time_ms']:.2f}ms")
        print(f"  操作数/秒: {result['operations_per_second']:.0f}")
        print()

def run_floating_point_advanced():
    """运行高级浮点数分析"""
    print("=" * 60)
    print("高级浮点数分析")
    print("=" * 60)
    
    # 创建演示实例（自动选择最佳设备）
    demo = FloatingPointDemo(device='auto')
    
    # 运行基本演示
    demo.run_complete_demo()
    
    # 运行多维度分析
    print("\n" + "=" * 40)
    multi_dim_results = demo.multi_dimensional_analysis()
    
    # 运行精度分析
    print("\n" + "=" * 40)
    precision_results = demo.precision_analysis()
    
    # 运行设备对比
    print("\n" + "=" * 40)
    device_results = demo.device_comparison()
    
    return {
        'multi_dimensional': multi_dim_results,
        'precision': precision_results,
        'device_comparison': device_results
    }

def run_attention_advanced():
    """运行高级注意力机制分析"""
    print("=" * 60)
    print("高级注意力机制分析")
    print("=" * 60)
    
    # 创建演示实例（自动选择最佳设备）
    demo = AttentionNondeterminismDemo(device='auto')
    
    # 运行完整演示
    demo.run_complete_demo()
    
    return demo.results

def run_batch_invariant_advanced():
    """运行高级批处理不变性分析"""
    print("=" * 60)
    print("高级批处理不变性分析")
    print("=" * 60)
    
    # 创建演示实例（自动选择最佳设备）
    demo = BatchInvariantDemo(device='auto')
    
    # 运行完整演示
    demo.run_complete_demo()
    
    return demo.results

def run_advanced_analysis():
    """运行高级分析"""
    print("=" * 60)
    print("高级综合分析")
    print("=" * 60)
    
    # 创建高级分析实例（自动选择最佳设备）
    analysis = AdvancedAnalysis(device='auto')
    
    # 运行综合分析报告
    results = analysis.create_comprehensive_report()
    
    return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='高级演示脚本 - 支持Mac MPS计算')
    parser.add_argument('--demo', type=str, default='all',
                       choices=['device', 'floating_point', 'attention', 'batch_invariant', 'advanced', 'all'],
                       help='选择要运行的演示')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'mps', 'auto'],
                       help='指定计算设备')
    
    args = parser.parse_args()
    
    print("🚀 启动高级演示脚本")
    print(f"选择演示: {args.demo}")
    print(f"计算设备: {args.device}")
    print()
    
    try:
        if args.demo in ['device', 'all']:
            run_device_benchmark()
        
        if args.demo in ['floating_point', 'all']:
            floating_point_results = run_floating_point_advanced()
        
        if args.demo in ['attention', 'all']:
            attention_results = run_attention_advanced()
        
        if args.demo in ['batch_invariant', 'all']:
            batch_invariant_results = run_batch_invariant_advanced()
        
        if args.demo in ['advanced', 'all']:
            advanced_results = run_advanced_analysis()
        
        print("\n" + "=" * 60)
        print("🎉 所有演示完成！")
        print("=" * 60)
        print("结果已保存到 experiments/plots/ 目录")
        print("可以查看以下文件：")
        print("- 设备基准测试结果")
        print("- 高级浮点数分析图表")
        print("- 注意力机制性能对比")
        print("- 批处理不变性分析")
        print("- 综合分析报告")
        
    except Exception as e:
        print(f"❌ 演示运行出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
