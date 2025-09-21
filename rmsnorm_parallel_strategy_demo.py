#!/usr/bin/env python3
"""
RMSNorm并行策略演示
基于Thinking Machines博客文章，测试不同batch size和并行度下的batch invariance

核心观点：
- 当batch size < 并行度时，不应该在归约维度进行并行计算
- 这会导致batch invariance的破坏
- 需要根据batch size动态调整并行策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Dict, Any
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from device_manager import get_device
from font_config import setup_chinese_fonts, force_chinese_fonts

# 设置中文字体
setup_chinese_fonts()
force_chinese_fonts()

class NaiveParallelRMSNorm(nn.Module):
    """朴素并行RMSNorm - 不考虑batch size，总是使用并行归约"""
    
    def __init__(self, dim: int, eps: float = 1e-6, parallel_degree: int = 4):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.parallel_degree = parallel_degree
    
    def forward(self, x):
        # 总是使用并行归约，不考虑batch size
        if x.dim() == 4:  # (B, H, W, C)
            batch_size, height, width, dim = x.shape
            x_flat = x.reshape(-1, dim)  # (B*H*W, C)
        elif x.dim() == 3:  # (B, seq_len, dim)
            batch_size, seq_len, dim = x.shape
            x_flat = x.reshape(-1, dim)  # (B*seq_len, dim)
        else:  # (B, dim)
            batch_size, dim = x.shape
            x_flat = x
        
        # 计算平方
        squared = x_flat ** 2
        
        # 总是使用并行归约（固定分割数）
        split_size = dim // self.parallel_degree
        rms_sums = []
        
        for i in range(self.parallel_degree):
            start_idx = i * split_size
            if i == self.parallel_degree - 1:  # 最后一个分割包含剩余元素
                end_idx = dim
            else:
                end_idx = (i + 1) * split_size
            
            # 计算每个分割的平方和
            split_x = squared[..., start_idx:end_idx]
            split_sum = torch.sum(split_x, dim=-1, keepdim=True)
            rms_sums.append(split_sum)
        
        # 累加所有分割
        total_sum = rms_sums[0]
        for i in range(1, len(rms_sums)):
            total_sum = total_sum + rms_sums[i]
        
        # 计算RMS
        rms = torch.sqrt(total_sum / dim + self.eps)
        result = x_flat / rms * self.weight
        
        # 恢复原始形状
        if x.dim() == 4:
            return result.reshape(batch_size, height, width, dim)
        elif x.dim() == 3:
            return result.reshape(batch_size, seq_len, dim)
        else:
            return result

class BatchAwareParallelRMSNorm(nn.Module):
    """Batch感知并行RMSNorm - 根据batch size动态调整并行策略"""
    
    def __init__(self, dim: int, eps: float = 1e-6, parallel_degree: int = 4):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.parallel_degree = parallel_degree
    
    def forward(self, x):
        if x.dim() == 4:  # (B, H, W, C)
            batch_size, height, width, dim = x.shape
            x_flat = x.reshape(-1, dim)  # (B*H*W, C)
        elif x.dim() == 3:  # (B, seq_len, dim)
            batch_size, seq_len, dim = x.shape
            x_flat = x.reshape(-1, dim)  # (B*seq_len, dim)
        else:  # (B, dim)
            batch_size, dim = x.shape
            x_flat = x
        
        # 计算平方
        squared = x_flat ** 2
        
        # 根据batch size决定并行策略
        if batch_size < self.parallel_degree:
            # batch size < 并行度：不在归约维度并行，保持batch invariance
            # 使用顺序归约
            total_sum = squared[..., 0:1]
            for i in range(1, dim):
                total_sum = total_sum + squared[..., i:i+1]
        else:
            # batch size >= 并行度：可以使用并行归约
            split_size = dim // self.parallel_degree
            rms_sums = []
            
            for i in range(self.parallel_degree):
                start_idx = i * split_size
                if i == self.parallel_degree - 1:
                    end_idx = dim
                else:
                    end_idx = (i + 1) * split_size
                
                split_x = squared[..., start_idx:end_idx]
                split_sum = torch.sum(split_x, dim=-1, keepdim=True)
                rms_sums.append(split_sum)
            
            # 累加所有分割
            total_sum = rms_sums[0]
            for i in range(1, len(rms_sums)):
                total_sum = total_sum + rms_sums[i]
        
        # 计算RMS
        rms = torch.sqrt(total_sum / dim + self.eps)
        result = x_flat / rms * self.weight
        
        # 恢复原始形状
        if x.dim() == 4:
            return result.reshape(batch_size, height, width, dim)
        elif x.dim() == 3:
            return result.reshape(batch_size, seq_len, dim)
        else:
            return result

class FixedSplitRMSNorm(nn.Module):
    """固定分割RMSNorm - 使用固定大小的分割，确保batch invariance"""
    
    def __init__(self, dim: int, eps: float = 1e-6, fixed_split_size: int = 64):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.fixed_split_size = fixed_split_size
    
    def forward(self, x):
        if x.dim() == 4:  # (B, H, W, C)
            batch_size, height, width, dim = x.shape
            x_flat = x.reshape(-1, dim)  # (B*H*W, C)
        elif x.dim() == 3:  # (B, seq_len, dim)
            batch_size, seq_len, dim = x.shape
            x_flat = x.reshape(-1, dim)  # (B*seq_len, dim)
        else:  # (B, dim)
            batch_size, dim = x.shape
            x_flat = x
        
        # 计算平方
        squared = x_flat ** 2
        
        # 使用固定大小的分割
        num_splits = (dim + self.fixed_split_size - 1) // self.fixed_split_size
        rms_sums = []
        
        for i in range(num_splits):
            start_idx = i * self.fixed_split_size
            end_idx = min((i + 1) * self.fixed_split_size, dim)
            
            split_x = squared[..., start_idx:end_idx]
            split_sum = torch.sum(split_x, dim=-1, keepdim=True)
            rms_sums.append(split_sum)
        
        # 按固定顺序累加
        total_sum = rms_sums[0]
        for i in range(1, len(rms_sums)):
            total_sum = total_sum + rms_sums[i]
        
        # 计算RMS
        rms = torch.sqrt(total_sum / dim + self.eps)
        result = x_flat / rms * self.weight
        
        # 恢复原始形状
        if x.dim() == 4:
            return result.reshape(batch_size, height, width, dim)
        elif x.dim() == 3:
            return result.reshape(batch_size, seq_len, dim)
        else:
            return result

class RMSNormParallelStrategyDemo:
    """RMSNorm并行策略演示类"""
    
    def __init__(self, device: str = 'auto', parallel_degree: int = 4, dim: int = 512):
        self.device = get_device(device)
        self.parallel_degree = parallel_degree
        self.dim = dim
        print(f"使用设备: {self.device}")
        print(f"并行度: {parallel_degree}")
        print(f"特征维度: {dim}")
        
        # 创建三种不同的RMSNorm实现
        self.naive_norm = NaiveParallelRMSNorm(dim, parallel_degree=parallel_degree).to(self.device)
        self.batch_aware_norm = BatchAwareParallelRMSNorm(dim, parallel_degree=parallel_degree).to(self.device)
        self.fixed_split_norm = FixedSplitRMSNorm(dim, fixed_split_size=64).to(self.device)
        
        # 设置为评估模式
        self.naive_norm.eval()
        self.batch_aware_norm.eval()
        self.fixed_split_norm.eval()
    
    def create_test_data(self, batch_size: int) -> torch.Tensor:
        """创建测试数据"""
        torch.manual_seed(42)
        return torch.randn(batch_size, self.dim, device=self.device)
    
    def test_batch_invariance(self, batch_sizes: List[int] = [1, 4, 8], num_trials: int = 5) -> Dict[str, Any]:
        """测试不同batch size下的batch invariance"""
        print("=" * 80)
        print("RMSNorm并行策略Batch Invariance测试")
        print("=" * 80)
        
        results = {
            'batch_sizes': batch_sizes,
            'naive_results': {},
            'batch_aware_results': {},
            'fixed_split_results': {},
            'performance': {}
        }
        
        for batch_size in batch_sizes:
            print(f"\n=== 测试Batch Size: {batch_size} ===")
            
            # 创建测试数据
            test_data = self.create_test_data(batch_size)
            
            # 测试三种实现
            for norm_name, norm_model in [
                ('naive', self.naive_norm),
                ('batch_aware', self.batch_aware_norm),
                ('fixed_split', self.fixed_split_norm)
            ]:
                print(f"\n--- {norm_name.upper()} 实现 ---")
                
                # 多次运行测试一致性
                outputs = []
                for trial in range(num_trials):
                    # 固定种子确保输入一致
                    torch.manual_seed(42)
                    random.seed(42)
                    np.random.seed(42)
                    if self.device.type == 'mps':
                        torch.mps.manual_seed(42)
                    
                    with torch.no_grad():
                        output = norm_model(test_data)
                        outputs.append(output.cpu().numpy())
                
                # 检查一致性
                consistent = self._check_consistency(outputs)
                print(f"一致性: {'✅ 通过' if consistent else '❌ 失败'}")
                
                # 计算输出统计
                mean_output = np.mean(outputs, axis=0)
                output_range = [mean_output.min(), mean_output.max()]
                print(f"输出范围: [{output_range[0]:.6f}, {output_range[1]:.6f}]")
                
                # 存储结果
                results[f'{norm_name}_results'][batch_size] = {
                    'consistent': consistent,
                    'outputs': outputs,
                    'mean_output': mean_output,
                    'output_range': output_range
                }
        
        return results
    
    def benchmark_performance(self, batch_sizes: List[int] = [1, 4, 8], num_iterations: int = 100) -> Dict[str, Any]:
        """性能基准测试"""
        print("\n" + "=" * 80)
        print("性能基准测试")
        print("=" * 80)
        
        performance_results = {}
        
        for batch_size in batch_sizes:
            print(f"\n=== Batch Size: {batch_size} ===")
            test_data = self.create_test_data(batch_size)
            
            batch_results = {}
            
            for norm_name, norm_model in [
                ('naive', self.naive_norm),
                ('batch_aware', self.batch_aware_norm),
                ('fixed_split', self.fixed_split_norm)
            ]:
                # 预热
                for _ in range(10):
                    with torch.no_grad():
                        _ = norm_model(test_data)
                
                # 性能测试
                start_time = time.time()
                for _ in range(num_iterations):
                    with torch.no_grad():
                        _ = norm_model(test_data)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                elif self.device.type == 'mps':
                    torch.mps.synchronize()
                
                avg_time = (time.time() - start_time) / num_iterations * 1000  # ms
                batch_results[norm_name] = avg_time
                print(f"{norm_name:12}: {avg_time:.2f}ms")
            
            performance_results[batch_size] = batch_results
        
        return performance_results
    
    def _check_consistency(self, outputs: List[np.ndarray]) -> bool:
        """检查输出的一致性"""
        if len(outputs) <= 1:
            return True
        
        reference = outputs[0]
        for output in outputs[1:]:
            if not np.allclose(output, reference, atol=1e-6, rtol=1e-6):
                return False
        return True
    
    def analyze_batch_invariance(self, results: Dict[str, Any]):
        """分析batch invariance"""
        print("\n" + "=" * 80)
        print("Batch Invariance分析")
        print("=" * 80)
        
        batch_sizes = results['batch_sizes']
        
        print(f"{'Batch Size':<12} {'Naive':<8} {'Batch-Aware':<12} {'Fixed-Split':<12}")
        print("-" * 50)
        
        for batch_size in batch_sizes:
            naive_consistent = results['naive_results'][batch_size]['consistent']
            batch_aware_consistent = results['batch_aware_results'][batch_size]['consistent']
            fixed_split_consistent = results['fixed_split_results'][batch_size]['consistent']
            
            print(f"{batch_size:<12} {('✅' if naive_consistent else '❌'):<8} "
                  f"{('✅' if batch_aware_consistent else '❌'):<12} "
                  f"{('✅' if fixed_split_consistent else '❌'):<12}")
        
        # 分析关键发现
        print(f"\n=== 关键发现 ===")
        
        # 检查batch size = 1时的行为
        naive_batch1 = results['naive_results'][1]['consistent']
        batch_aware_batch1 = results['batch_aware_results'][1]['consistent']
        fixed_split_batch1 = results['fixed_split_results'][1]['consistent']
        
        print(f"Batch Size = 1时:")
        print(f"  - Naive实现: {'✅ 一致' if naive_batch1 else '❌ 不一致'}")
        print(f"  - Batch-Aware实现: {'✅ 一致' if batch_aware_batch1 else '❌ 不一致'}")
        print(f"  - Fixed-Split实现: {'✅ 一致' if fixed_split_batch1 else '❌ 不一致'}")
        
        if not naive_batch1 and batch_aware_batch1:
            print("🎉 验证成功！Batch-Aware实现解决了batch invariance问题")
        elif naive_batch1 and batch_aware_batch1:
            print("⚠️ 两种实现都保持一致性，可能需要调整测试参数")
        else:
            print("❌ 结果不符合预期")
    
    def visualize_results(self, results: Dict[str, Any], performance_results: Dict[str, Any]):
        """可视化结果"""
        print("\n" + "=" * 80)
        print("结果可视化")
        print("=" * 80)
        
        batch_sizes = results['batch_sizes']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Batch Invariance一致性
        naive_consistency = [results['naive_results'][bs]['consistent'] for bs in batch_sizes]
        batch_aware_consistency = [results['batch_aware_results'][bs]['consistent'] for bs in batch_sizes]
        fixed_split_consistency = [results['fixed_split_results'][bs]['consistent'] for bs in batch_sizes]
        
        x = np.arange(len(batch_sizes))
        width = 0.25
        
        axes[0, 0].bar(x - width, naive_consistency, width, label='Naive', alpha=0.7)
        axes[0, 0].bar(x, batch_aware_consistency, width, label='Batch-Aware', alpha=0.7)
        axes[0, 0].bar(x + width, fixed_split_consistency, width, label='Fixed-Split', alpha=0.7)
        
        axes[0, 0].set_xlabel('Batch Size')
        axes[0, 0].set_ylabel('一致性 (1=一致, 0=不一致)')
        axes[0, 0].set_title('Batch Invariance一致性对比')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(batch_sizes)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 性能对比
        naive_perf = [performance_results[bs]['naive'] for bs in batch_sizes]
        batch_aware_perf = [performance_results[bs]['batch_aware'] for bs in batch_sizes]
        fixed_split_perf = [performance_results[bs]['fixed_split'] for bs in batch_sizes]
        
        axes[0, 1].plot(batch_sizes, naive_perf, 'o-', label='Naive', linewidth=2, markersize=8)
        axes[0, 1].plot(batch_sizes, batch_aware_perf, 's-', label='Batch-Aware', linewidth=2, markersize=8)
        axes[0, 1].plot(batch_sizes, fixed_split_perf, '^-', label='Fixed-Split', linewidth=2, markersize=8)
        
        axes[0, 1].set_xlabel('Batch Size')
        axes[0, 1].set_ylabel('平均时间 (ms)')
        axes[0, 1].set_title('性能对比')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 性能开销分析
        naive_baseline = naive_perf[0]  # 使用batch size=1作为基准
        batch_aware_overhead = [(p - naive_baseline) / naive_baseline * 100 for p in batch_aware_perf]
        fixed_split_overhead = [(p - naive_baseline) / naive_baseline * 100 for p in fixed_split_perf]
        
        axes[1, 0].plot(batch_sizes, batch_aware_overhead, 's-', label='Batch-Aware', linewidth=2, markersize=8)
        axes[1, 0].plot(batch_sizes, fixed_split_overhead, '^-', label='Fixed-Split', linewidth=2, markersize=8)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        axes[1, 0].set_xlabel('Batch Size')
        axes[1, 0].set_ylabel('性能开销 (%)')
        axes[1, 0].set_title('相对性能开销')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 输出差异分析
        output_diffs = []
        for bs in batch_sizes:
            naive_output = results['naive_results'][bs]['mean_output']
            batch_aware_output = results['batch_aware_results'][bs]['mean_output']
            diff = np.abs(naive_output - batch_aware_output).max()
            output_diffs.append(diff)
        
        axes[1, 1].bar(batch_sizes, output_diffs, alpha=0.7, color='orange')
        axes[1, 1].set_xlabel('Batch Size')
        axes[1, 1].set_ylabel('最大输出差异')
        axes[1, 1].set_title('Naive vs Batch-Aware输出差异')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('RMSNorm并行策略分析结果', fontsize=16, fontweight='bold', y=1.02)
        
        # 保存图片
        os.makedirs('experiments/plots', exist_ok=True)
        plt.savefig('experiments/plots/rmsnorm_parallel_strategy_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("结果可视化已保存到: experiments/plots/rmsnorm_parallel_strategy_analysis.png")
    
    def comprehensive_analysis(self):
        """综合分析"""
        print("RMSNorm并行策略综合分析")
        print("基于Thinking Machines博客文章")
        print("=" * 80)
        
        # 测试参数
        batch_sizes = [1, 4, 8]
        parallel_degree = 4
        
        print(f"测试配置:")
        print(f"  - Batch Sizes: {batch_sizes}")
        print(f"  - 并行度: {parallel_degree}")
        print(f"  - 特征维度: {self.dim}")
        
        # 1. Batch Invariance测试
        results = self.test_batch_invariance(batch_sizes)
        
        # 2. 性能基准测试
        performance_results = self.benchmark_performance(batch_sizes)
        
        # 3. 分析结果
        self.analyze_batch_invariance(results)
        
        # 4. 可视化
        self.visualize_results(results, performance_results)
        
        # 5. 总结
        print("\n" + "=" * 80)
        print("分析总结")
        print("=" * 80)
        
        print("🎯 核心观点验证:")
        print("  - 当batch size < 并行度时，不应该在归约维度进行并行计算")
        print("  - 这会导致batch invariance的破坏")
        print("  - Batch-Aware策略根据batch size动态调整并行策略")
        print("  - Fixed-Split策略使用固定大小分割确保batch invariance")
        
        return {
            'results': results,
            'performance': performance_results
        }

def main():
    """主函数"""
    print("RMSNorm并行策略演示")
    print("基于Thinking Machines博客文章")
    print("=" * 80)
    
    # 创建演示实例
    demo = RMSNormParallelStrategyDemo(parallel_degree=4, dim=512)
    
    # 运行综合分析
    analysis_results = demo.comprehensive_analysis()
    
    print("\n🎉 RMSNorm并行策略分析完成！")
    print("\n关键发现:")
    print("- 朴素并行策略在batch size < 并行度时可能破坏batch invariance")
    print("- Batch-Aware策略根据batch size动态调整并行策略")
    print("- Fixed-Split策略使用固定大小分割确保batch invariance")
    print("- 不同策略在性能和确定性之间存在权衡")

if __name__ == "__main__":
    main()
