"""
高级分析模块

这个模块提供了更深入的分析功能，包括多维度验证、性能优化和高级指标分析。
特别针对Mac的MPS计算进行了优化。
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
import time
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 导入字体配置和设备管理
try:
    from .font_config import setup_chinese_fonts, force_chinese_fonts
    from .device_manager import get_device, device_manager, benchmark_devices
except ImportError:
    from font_config import setup_chinese_fonts, force_chinese_fonts
    from device_manager import get_device, device_manager, benchmark_devices

# 设置中文字体
setup_chinese_fonts()

class AdvancedAnalysis:
    """高级分析类"""
    
    def __init__(self, device: str = 'auto'):
        """
        初始化高级分析
        
        Args:
            device: 计算设备 ('cpu', 'cuda', 'mps', 'auto')
        """
        if device == 'auto':
            self.device = get_device()
        else:
            self.device = get_device(device)
        
        self.results = {}
        self.device_info = device_manager.get_memory_info(self.device.type)
        
        print(f"高级分析使用设备: {self.device}")
        if self.device.type == 'mps':
            print("使用Apple Silicon MPS加速")
    
    def comprehensive_benchmark(self, 
                              matrix_sizes: List[int] = [64, 128, 256, 512, 1024],
                              operations: List[str] = ['add', 'multiply', 'matmul', 'attention']) -> Dict[str, Any]:
        """综合基准测试"""
        print("=== 综合基准测试 ===")
        
        results = {}
        
        for size in matrix_sizes:
            print(f"测试矩阵大小: {size}x{size}")
            results[size] = {}
            
            for op in operations:
                print(f"  测试操作: {op}")
                
                # 创建测试数据（MPS只支持float32）
                dtype = torch.float32 if self.device.type == 'mps' else torch.float32
                a = torch.randn(size, size, device=self.device, dtype=dtype)
                b = torch.randn(size, size, device=self.device, dtype=dtype)
                
                # 预热
                for _ in range(5):
                    if op == 'add':
                        _ = a + b
                    elif op == 'multiply':
                        _ = a * b
                    elif op == 'matmul':
                        _ = torch.matmul(a, b)
                    elif op == 'attention':
                        _ = torch.matmul(a, b.transpose(-2, -1))
                
                # 同步设备
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                elif self.device.type == 'mps':
                    torch.mps.synchronize()
                
                # 性能测试
                start_time = time.time()
                for _ in range(10):
                    if op == 'add':
                        result = a + b
                    elif op == 'multiply':
                        result = a * b
                    elif op == 'matmul':
                        result = torch.matmul(a, b)
                    elif op == 'attention':
                        result = torch.matmul(a, b.transpose(-2, -1))
                
                # 同步设备
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                elif self.device.type == 'mps':
                    torch.mps.synchronize()
                
                end_time = time.time()
                avg_time = (end_time - start_time) / 10
                
                results[size][op] = {
                    'avg_time_ms': avg_time * 1000,
                    'operations_per_second': (size ** 3) / avg_time if op in ['matmul', 'attention'] else (size ** 2) / avg_time
                }
                
                print(f"    平均时间: {avg_time*1000:.2f}ms")
        
        return results
    
    def memory_usage_analysis(self, max_size: int = 2048) -> Dict[str, Any]:
        """内存使用分析"""
        print("=== 内存使用分析 ===")
        
        results = {}
        sizes = [64, 128, 256, 512, 1024, 2048]
        
        for size in sizes:
            if size > max_size:
                break
                
            print(f"测试矩阵大小: {size}x{size}")
            
            # 获取初始内存
            initial_memory = self._get_memory_usage()
            
            # 创建大矩阵
            a = torch.randn(size, size, device=self.device, dtype=torch.float32)
            b = torch.randn(size, size, device=self.device, dtype=torch.float32)
            
            # 获取创建后内存
            after_creation_memory = self._get_memory_usage()
            
            # 执行操作
            result = torch.matmul(a, b)
            
            # 获取操作后内存
            after_operation_memory = self._get_memory_usage()
            
            # 清理
            del a, b, result
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            elif self.device.type == 'mps':
                torch.mps.empty_cache()
            
            results[size] = {
                'initial_memory_mb': initial_memory,
                'after_creation_memory_mb': after_creation_memory,
                'after_operation_memory_mb': after_operation_memory,
                'memory_increase_mb': after_creation_memory - initial_memory,
                'operation_memory_mb': after_operation_memory - after_creation_memory
            }
            
            print(f"  内存增加: {after_creation_memory - initial_memory:.2f}MB")
            print(f"  操作内存: {after_operation_memory - after_creation_memory:.2f}MB")
        
        return results
    
    def _get_memory_usage(self) -> float:
        """获取内存使用量（MB）"""
        if self.device.type == 'cuda':
            return torch.cuda.memory_allocated() / (1024 * 1024)
        elif self.device.type == 'mps':
            # MPS不提供详细内存信息，使用系统内存
            import psutil
            return (psutil.virtual_memory().total - psutil.virtual_memory().available) / (1024 * 1024)
        else:
            import psutil
            return (psutil.virtual_memory().total - psutil.virtual_memory().available) / (1024 * 1024)
    
    def numerical_stability_analysis(self, 
                                   dimensions: List[int] = [64, 128, 256, 512],
                                   iterations: int = 100) -> Dict[str, Any]:
        """数值稳定性分析"""
        print("=== 数值稳定性分析 ===")
        
        results = {}
        
        for dim in dimensions:
            print(f"测试维度: {dim}x{dim}")
            
            differences = []
            computation_times = []
            
            for i in range(iterations):
                # 创建随机矩阵
                a = torch.randn(dim, dim, device=self.device, dtype=torch.float32)
                b = torch.randn(dim, dim, device=self.device, dtype=torch.float32)
                c = torch.randn(dim, dim, device=self.device, dtype=torch.float32)
                
                # 测试不同计算顺序
                start_time = time.time()
                
                result1 = (a + b) + c
                result2 = a + (b + c)
                
                # 同步设备
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                elif self.device.type == 'mps':
                    torch.mps.synchronize()
                
                end_time = time.time()
                
                # 计算差异
                difference = torch.max(torch.abs(result1 - result2)).item()
                differences.append(difference)
                computation_times.append(end_time - start_time)
            
            # 统计分析
            differences = np.array(differences)
            computation_times = np.array(computation_times)
            
            results[dim] = {
                'mean_difference': np.mean(differences),
                'std_difference': np.std(differences),
                'max_difference': np.max(differences),
                'min_difference': np.min(differences),
                'median_difference': np.median(differences),
                'mean_computation_time': np.mean(computation_times),
                'std_computation_time': np.std(computation_times),
                'iterations': iterations
            }
            
            print(f"  平均差异: {np.mean(differences):.2e}")
            print(f"  标准差: {np.std(differences):.2e}")
            print(f"  最大差异: {np.max(differences):.2e}")
            print(f"  平均计算时间: {np.mean(computation_times)*1000:.2f}ms")
        
        return results
    
    def device_comparison_analysis(self) -> Dict[str, Any]:
        """设备对比分析"""
        print("=== 设备对比分析 ===")
        
        # 获取所有可用设备
        available_devices = ['cpu']
        if torch.cuda.is_available():
            available_devices.append('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            available_devices.append('mps')
        
        results = {}
        test_size = 512
        
        for device_name in available_devices:
            print(f"测试设备: {device_name}")
            
            device = get_device(device_name)
            
            # 创建测试数据
            a = torch.randn(test_size, test_size, device=device, dtype=torch.float32)
            b = torch.randn(test_size, test_size, device=device, dtype=torch.float32)
            
            # 预热
            for _ in range(5):
                _ = torch.matmul(a, b)
            
            # 同步设备
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()
            
            # 性能测试
            start_time = time.time()
            for _ in range(20):
                result = torch.matmul(a, b)
            end_time = time.time()
            
            # 同步设备
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()
            
            avg_time = (end_time - start_time) / 20
            
            # 内存使用
            memory_usage = self._get_memory_usage_for_device(device)
            
            results[device_name] = {
                'avg_time_ms': avg_time * 1000,
                'operations_per_second': (test_size ** 3) / avg_time,
                'memory_usage_mb': memory_usage,
                'device': device_name
            }
            
            print(f"  平均时间: {avg_time*1000:.2f}ms")
            print(f"  内存使用: {memory_usage:.2f}MB")
        
        return results
    
    def _get_memory_usage_for_device(self, device: torch.device) -> float:
        """获取特定设备的内存使用量"""
        if device.type == 'cuda':
            return torch.cuda.memory_allocated() / (1024 * 1024)
        elif device.type == 'mps':
            import psutil
            return (psutil.virtual_memory().total - psutil.virtual_memory().available) / (1024 * 1024)
        else:
            import psutil
            return (psutil.virtual_memory().total - psutil.virtual_memory().available) / (1024 * 1024)
    
    def create_comprehensive_report(self, save_path: str = 'experiments/plots/advanced_analysis_report.png') -> None:
        """创建综合分析报告"""
        print("=== 创建综合分析报告 ===")
        
        # 确保中文字体设置
        force_chinese_fonts()
        
        # 运行所有分析
        benchmark_results = self.comprehensive_benchmark()
        memory_results = self.memory_usage_analysis()
        stability_results = self.numerical_stability_analysis()
        device_results = self.device_comparison_analysis()
        
        # 创建综合图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('高级分析综合报告', fontsize=20, fontweight='bold')
        
        # 子图1: 性能基准测试
        ax1 = axes[0, 0]
        sizes = list(benchmark_results.keys())
        matmul_times = [benchmark_results[size]['matmul']['avg_time_ms'] for size in sizes]
        ax1.plot(sizes, matmul_times, 'o-', linewidth=2, markersize=8, label='矩阵乘法')
        ax1.set_xlabel('矩阵大小')
        ax1.set_ylabel('平均时间 (ms)')
        ax1.set_title('性能基准测试')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 子图2: 内存使用分析
        ax2 = axes[0, 1]
        memory_sizes = list(memory_results.keys())
        memory_usage = [memory_results[size]['memory_increase_mb'] for size in memory_sizes]
        ax2.plot(memory_sizes, memory_usage, 's-', linewidth=2, markersize=8, color='red')
        ax2.set_xlabel('矩阵大小')
        ax2.set_ylabel('内存使用 (MB)')
        ax2.set_title('内存使用分析')
        ax2.grid(True, alpha=0.3)
        
        # 子图3: 数值稳定性
        ax3 = axes[0, 2]
        stability_sizes = list(stability_results.keys())
        mean_diffs = [stability_results[size]['mean_difference'] for size in stability_sizes]
        ax3.semilogy(stability_sizes, mean_diffs, '^-', linewidth=2, markersize=8, color='green')
        ax3.set_xlabel('矩阵大小')
        ax3.set_ylabel('平均差异 (log scale)')
        ax3.set_title('数值稳定性分析')
        ax3.grid(True, alpha=0.3)
        
        # 子图4: 设备对比
        ax4 = axes[1, 0]
        device_names = list(device_results.keys())
        device_times = [device_results[device]['avg_time_ms'] for device in device_names]
        bars = ax4.bar(device_names, device_times, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax4.set_ylabel('平均时间 (ms)')
        ax4.set_title('设备性能对比')
        ax4.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, time in zip(bars, device_times):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{time:.1f}ms', ha='center', va='bottom', fontsize=10)
        
        # 子图5: 操作类型对比
        ax5 = axes[1, 1]
        operations = ['add', 'multiply', 'matmul', 'attention']
        op_times = [benchmark_results[512][op]['avg_time_ms'] for op in operations]
        bars = ax5.bar(operations, op_times, color=['#d62728', '#9467bd', '#8c564b', '#e377c2'])
        ax5.set_ylabel('平均时间 (ms)')
        ax5.set_title('操作类型对比 (512x512)')
        ax5.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, time in zip(bars, op_times):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{time:.1f}ms', ha='center', va='bottom', fontsize=10)
        
        # 子图6: 统计信息
        ax6 = axes[1, 2]
        stats_text = f"""
        设备信息:
        当前设备: {self.device}
        系统: {device_manager.device_info['system']}
        CPU核心: {device_manager.device_info['cpu_count']}
        总内存: {device_manager.device_info['memory_total']/(1024**3):.1f}GB
        
        测试结果:
        最大矩阵: {max(sizes)}x{max(sizes)}
        最高性能: {max([device_results[d]['operations_per_second'] for d in device_results]):.0f} ops/s
        最小差异: {min([stability_results[s]['mean_difference'] for s in stability_results]):.2e}
        """
        ax6.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax6.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"综合分析报告已保存到: {save_path}")
        
        # 保存详细结果
        self.results = {
            'benchmark': benchmark_results,
            'memory': memory_results,
            'stability': stability_results,
            'device_comparison': device_results
        }
        
        return self.results
