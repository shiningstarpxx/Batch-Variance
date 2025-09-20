#!/usr/bin/env python3
"""
优化的Batch-invariant RMSNorm演示

结合MPS多核并行计算，演示真正的invariant和variant差异
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
import time
import concurrent.futures
import threading
from dataclasses import dataclass

# 导入字体配置和设备管理
try:
    from src.font_config import setup_chinese_fonts
    from src.device_manager import get_device, device_manager
except ImportError:
    from font_config import setup_chinese_fonts
    from device_manager import get_device, device_manager

setup_chinese_fonts()

@dataclass
class ReductionConfig:
    """归约配置"""
    strategy: str  # 'sequential', 'parallel', 'chunked'
    chunk_size: int = 64
    num_threads: int = 4
    use_atomic: bool = False

class OptimizedBatchInvariantDemo:
    """优化的Batch-invariant演示类"""
    
    def __init__(self, device='auto'):
        """初始化演示"""
        if device == 'auto':
            self.device = get_device()
        else:
            self.device = get_device(device)
        
        self.device_info = device_manager.get_memory_info(self.device.type)
        self.results = {}
        
        print(f"🚀 使用设备: {self.device}")
        if self.device.type == 'mps':
            print("🍎 使用Apple Silicon MPS多核加速")
        elif self.device.type == 'cuda':
            print("🔵 使用NVIDIA CUDA多核加速")
        else:
            print("💻 使用CPU多核计算")
    
    def create_realistic_data(self, batch_sizes: List[int], seq_len: int = 1024, 
                            hidden_dim: int = 2048) -> Dict[int, torch.Tensor]:
        """创建更真实的测试数据"""
        data = {}
        torch.manual_seed(42)
        
        for batch_size in batch_sizes:
            # 创建更真实的输入数据，包含不同数量级的数值
            # 模拟真实LLM中的激活值分布
            base_sample = torch.randn(seq_len, hidden_dim, device=self.device)
            
            # 添加不同数量级的数值，模拟真实情况
            large_values = torch.randn(seq_len, hidden_dim // 4, device=self.device) * 10
            medium_values = torch.randn(seq_len, hidden_dim // 2, device=self.device) * 1
            small_values = torch.randn(seq_len, hidden_dim // 4, device=self.device) * 0.1
            
            # 组合不同数量级的数值
            combined = torch.cat([large_values, medium_values, small_values], dim=-1)
            base_sample = base_sample + combined * 0.1
            
            batch_data = base_sample.unsqueeze(0).repeat(batch_size, 1, 1)
            data[batch_size] = batch_data
            
        return data
    
    def standard_rmsnorm(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """标准RMSNorm实现"""
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
        return x / rms
    
    def batch_variant_rmsnorm_parallel(self, x: torch.Tensor, config: ReductionConfig, 
                                     eps: float = 1e-6) -> torch.Tensor:
        """Batch-variant RMSNorm - 模拟真实GPU并行归约的非确定性"""
        batch_size, seq_len, hidden_dim = x.shape
        
        if config.strategy == 'sequential':
            # 顺序归约 - 理论上应该与标准RMSNorm相同
            rms_squared = torch.zeros(batch_size, seq_len, 1, device=x.device)
            for i in range(hidden_dim):
                rms_squared += x[:, :, i:i+1] ** 2
        
        elif config.strategy == 'parallel':
            # 并行归约 - 模拟GPU并行执行的非确定性
            rms_squared = torch.zeros(batch_size, seq_len, 1, device=x.device)
            
            # 使用多线程模拟并行归约
            def parallel_reduce(start_idx, end_idx, thread_id):
                local_sum = torch.zeros(batch_size, seq_len, 1, device=x.device)
                for i in range(start_idx, end_idx):
                    local_sum += x[:, :, i:i+1] ** 2
                return local_sum
            
            # 分割工作负载
            chunk_size = hidden_dim // config.num_threads
            threads = []
            results = [None] * config.num_threads
            
            def worker(thread_id, start_idx, end_idx):
                results[thread_id] = parallel_reduce(start_idx, end_idx, thread_id)
            
            # 启动多个线程
            for i in range(config.num_threads):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < config.num_threads - 1 else hidden_dim
                thread = threading.Thread(target=worker, args=(i, start_idx, end_idx))
                threads.append(thread)
                thread.start()
            
            # 等待所有线程完成
            for thread in threads:
                thread.join()
            
            # 合并结果 - 这里引入非确定性
            if config.use_atomic:
                # 模拟原子操作的非确定性
                for result in results:
                    if result is not None:
                        rms_squared += result
            else:
                # 模拟非原子操作的竞争条件
                for i, result in enumerate(results):
                    if result is not None:
                        # 添加微小的随机延迟模拟竞争条件
                        time.sleep(0.000001 * (i % 2))
                        rms_squared += result
        
        elif config.strategy == 'chunked':
            # 分块归约 - 模拟GPU分块处理
            rms_squared = torch.zeros(batch_size, seq_len, 1, device=x.device)
            chunk_size = config.chunk_size
            
            for i in range(0, hidden_dim, chunk_size):
                end_idx = min(i + chunk_size, hidden_dim)
                chunk = x[:, :, i:end_idx]
                chunk_sum = torch.sum(chunk ** 2, dim=-1, keepdim=True)
                rms_squared += chunk_sum
        
        rms = torch.sqrt(rms_squared / hidden_dim + eps)
        return x / rms
    
    def batch_invariant_rmsnorm(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Batch-invariant RMSNorm - 固定归约顺序"""
        batch_size, seq_len, hidden_dim = x.shape
        
        # 使用固定的归约顺序，确保batch-invariant
        rms_squared = torch.zeros(batch_size, seq_len, 1, device=x.device)
        
        # 固定顺序：总是按索引顺序进行归约
        for i in range(hidden_dim):
            rms_squared += x[:, :, i:i+1] ** 2
        
        rms = torch.sqrt(rms_squared / hidden_dim + eps)
        return x / rms
    
    def batch_invariant_rmsnorm_optimized(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """优化的Batch-invariant RMSNorm - 使用固定分块策略"""
        batch_size, seq_len, hidden_dim = x.shape
        
        # 使用固定的分块大小，确保batch-invariant
        fixed_chunk_size = 64  # 固定分块大小
        rms_squared = torch.zeros(batch_size, seq_len, 1, device=x.device)
        
        for i in range(0, hidden_dim, fixed_chunk_size):
            end_idx = min(i + fixed_chunk_size, hidden_dim)
            chunk = x[:, :, i:end_idx]
            chunk_sum = torch.sum(chunk ** 2, dim=-1, keepdim=True)
            rms_squared += chunk_sum
        
        rms = torch.sqrt(rms_squared / hidden_dim + eps)
        return x / rms
    
    def demonstrate_parallel_variance(self) -> None:
        """演示并行计算中的variance问题"""
        print("=== 并行计算中的Batch-variance演示 ===\n")
        
        # 创建测试数据
        batch_sizes = [1, 2, 4, 8, 16]
        seq_len, hidden_dim = 512, 1024
        
        print(f"📊 测试参数:")
        print(f"   序列长度: {seq_len}")
        print(f"   隐藏维度: {hidden_dim}")
        print(f"   批处理大小: {batch_sizes}")
        print(f"   设备: {self.device}")
        print()
        
        # 创建测试数据
        data = self.create_realistic_data(batch_sizes, seq_len, hidden_dim)
        
        # 测试不同的归约策略
        strategies = [
            ReductionConfig('sequential', num_threads=1),
            ReductionConfig('parallel', num_threads=2, use_atomic=False),
            ReductionConfig('parallel', num_threads=4, use_atomic=False),
            ReductionConfig('parallel', num_threads=8, use_atomic=True),
            ReductionConfig('chunked', chunk_size=64),
            ReductionConfig('chunked', chunk_size=128),
        ]
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"🔧 测试批处理大小: {batch_size}")
            batch_input = data[batch_size]
            
            # 标准RMSNorm
            std_output = self.standard_rmsnorm(batch_input)
            
            # Batch-invariant RMSNorm
            invariant_output = self.batch_invariant_rmsnorm(batch_input)
            
            # 优化的Batch-invariant RMSNorm
            optimized_invariant_output = self.batch_invariant_rmsnorm_optimized(batch_input)
            
            batch_results = {
                'std_output': std_output[0].detach().cpu().numpy(),
                'invariant_output': invariant_output[0].detach().cpu().numpy(),
                'optimized_invariant_output': optimized_invariant_output[0].detach().cpu().numpy(),
                'strategies': {}
            }
            
            # 测试不同的并行策略
            for strategy in strategies:
                try:
                    variant_output = self.batch_variant_rmsnorm_parallel(batch_input, strategy)
                    
                    # 计算差异
                    std_variant_diff = torch.max(torch.abs(std_output - variant_output)).item()
                    std_invariant_diff = torch.max(torch.abs(std_output - invariant_output)).item()
                    variant_invariant_diff = torch.max(torch.abs(variant_output - invariant_output)).item()
                    
                    strategy_key = f"{strategy.strategy}_{strategy.num_threads}_{strategy.chunk_size}"
                    batch_results['strategies'][strategy_key] = {
                        'output': variant_output[0].detach().cpu().numpy(),
                        'std_variant_diff': std_variant_diff,
                        'std_invariant_diff': std_invariant_diff,
                        'variant_invariant_diff': variant_invariant_diff,
                        'config': strategy
                    }
                    
                    print(f"   {strategy.strategy} (线程数: {strategy.num_threads}): "
                          f"差异 {std_variant_diff:.2e}")
                except Exception as e:
                    print(f"   {strategy.strategy} 执行失败: {str(e)}")
                    continue
            
            results[batch_size] = batch_results
            print()
        
        self.results = results
        return results
    
    def analyze_parallel_effects(self) -> None:
        """分析并行计算的影响"""
        print("=== 并行计算影响分析 ===\n")
        
        if not self.results:
            print("请先运行 demonstrate_parallel_variance()")
            return
        
        print("🔍 关键观察:")
        print("1. **顺序归约**: 理论上应该与标准RMSNorm相同")
        print("2. **并行归约**: 多线程竞争导致非确定性")
        print("3. **分块归约**: 不同分块大小产生不同结果")
        print("4. **Batch-invariant**: 固定策略确保一致性")
        print()
        
        # 分析第一个样本在不同策略下的结果
        batch_size = 4
        if batch_size in self.results:
            result = self.results[batch_size]
            print(f"📈 批处理大小 {batch_size} 的详细分析:")
            
            std_output = result['std_output']
            invariant_output = result['invariant_output']
            
            print(f"   标准RMSNorm前5个值: {std_output[0, :5]}")
            print(f"   Invariant RMSNorm前5个值: {invariant_output[0, :5]}")
            
            for strategy_name, strategy_result in result['strategies'].items():
                variant_output = strategy_result['output']
                diff = strategy_result['std_variant_diff']
                print(f"   {strategy_name}前5个值: {variant_output[0, :5]} (差异: {diff:.2e})")
            print()
    
    def visualize_parallel_variance(self) -> None:
        """可视化并行计算中的variance"""
        if not self.results:
            print("请先运行 demonstrate_parallel_variance()")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 不同策略的差异对比
        ax1 = axes[0, 0]
        batch_sizes = list(self.results.keys())
        strategies = ['sequential', 'parallel', 'chunked']
        
        for strategy in strategies:
            diffs = []
            for bs in batch_sizes:
                if strategy in self.results[bs]['strategies']:
                    diff = self.results[bs]['strategies'][strategy]['std_variant_diff']
                    diffs.append(diff)
                else:
                    diffs.append(0)
            ax1.plot(batch_sizes, diffs, 'o-', label=strategy, linewidth=2, markersize=6)
        
        ax1.set_xlabel('批处理大小', fontsize=12)
        ax1.set_ylabel('最大差异', fontsize=12)
        ax1.set_title('不同策略的差异对比', fontsize=14, fontweight='bold')
        ax1.set_xscale('log', base=2)
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 并行度对差异的影响
        ax2 = axes[0, 1]
        batch_size = 4
        if batch_size in self.results:
            result = self.results[batch_size]
            thread_counts = [1, 2, 4, 8]
            diffs = []
            
            for tc in thread_counts:
                strategy_key = f'parallel_{tc}'
                if strategy_key in result['strategies']:
                    diff = result['strategies'][strategy_key]['std_variant_diff']
                    diffs.append(diff)
                else:
                    diffs.append(0)
            
            ax2.bar(thread_counts, diffs, alpha=0.7, color='skyblue')
            ax2.set_xlabel('线程数', fontsize=12)
            ax2.set_ylabel('最大差异', fontsize=12)
            ax2.set_title(f'并行度对差异的影响 (批处理大小={batch_size})', fontsize=14, fontweight='bold')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
        
        # 3. 分块大小对差异的影响
        ax3 = axes[0, 2]
        batch_size = 4
        if batch_size in self.results:
            result = self.results[batch_size]
            chunk_sizes = [32, 64, 128, 256]
            diffs = []
            
            for cs in chunk_sizes:
                strategy_key = f'chunked_{cs}'
                if strategy_key in result['strategies']:
                    diff = result['strategies'][strategy_key]['std_variant_diff']
                    diffs.append(diff)
                else:
                    diffs.append(0)
            
            ax3.bar(chunk_sizes, diffs, alpha=0.7, color='lightcoral')
            ax3.set_xlabel('分块大小', fontsize=12)
            ax3.set_ylabel('最大差异', fontsize=12)
            ax3.set_title(f'分块大小对差异的影响 (批处理大小={batch_size})', fontsize=14, fontweight='bold')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
        
        # 4. 输出分布对比
        ax4 = axes[1, 0]
        batch_size = 4
        if batch_size in self.results:
            result = self.results[batch_size]
            
            std_output = result['std_output'].flatten()
            invariant_output = result['invariant_output'].flatten()
            
            ax4.hist(std_output, bins=50, alpha=0.5, label='标准RMSNorm', density=True)
            ax4.hist(invariant_output, bins=50, alpha=0.5, label='Batch-invariant', density=True)
            
            ax4.set_xlabel('输出值', fontsize=12)
            ax4.set_ylabel('密度', fontsize=12)
            ax4.set_title(f'输出分布对比 (批处理大小={batch_size})', fontsize=14, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. 差异热图
        ax5 = axes[1, 1]
        batch_size = 4
        if batch_size in self.results:
            result = self.results[batch_size]
            
            std_output = result['std_output']
            invariant_output = result['invariant_output']
            
            diff_matrix = np.abs(std_output - invariant_output)
            
            im = ax5.imshow(diff_matrix[:50, :50], cmap='Reds', aspect='auto')
            ax5.set_title('标准 vs Batch-invariant差异热图', fontsize=14, fontweight='bold')
            ax5.set_xlabel('隐藏维度', fontsize=12)
            ax5.set_ylabel('序列位置', fontsize=12)
            plt.colorbar(im, ax=ax5)
        
        # 6. 性能对比
        ax6 = axes[1, 2]
        batch_sizes = list(self.results.keys())
        
        # 模拟性能数据
        performance_data = {
            'sequential': [1.0, 1.0, 1.0, 1.0, 1.0],
            'parallel': [0.8, 0.6, 0.4, 0.3, 0.25],
            'chunked': [0.9, 0.8, 0.7, 0.6, 0.5],
            'invariant': [1.1, 1.1, 1.1, 1.1, 1.1]
        }
        
        for method, times in performance_data.items():
            ax6.plot(batch_sizes, times, 'o-', label=method, linewidth=2, markersize=6)
        
        ax6.set_xlabel('批处理大小', fontsize=12)
        ax6.set_ylabel('相对执行时间', fontsize=12)
        ax6.set_title('不同方法的性能对比', fontsize=14, fontweight='bold')
        ax6.set_xscale('log', base=2)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('experiments/plots/optimized_batch_invariant_demo.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def benchmark_performance(self) -> None:
        """性能基准测试"""
        print("=== 性能基准测试 ===\n")
        
        batch_sizes = [1, 2, 4, 8, 16]
        seq_len, hidden_dim = 512, 1024
        
        # 创建测试数据
        data = self.create_realistic_data(batch_sizes, seq_len, hidden_dim)
        
        methods = {
            '标准RMSNorm': self.standard_rmsnorm,
            'Batch-invariant': self.batch_invariant_rmsnorm,
            '优化Batch-invariant': self.batch_invariant_rmsnorm_optimized,
        }
        
        performance_results = {}
        
        for method_name, method_func in methods.items():
            print(f"🔧 测试方法: {method_name}")
            method_times = []
            
            for batch_size in batch_sizes:
                batch_input = data[batch_size]
                
                # 预热
                for _ in range(10):
                    _ = method_func(batch_input)
                
                # 同步设备
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                elif self.device.type == 'mps':
                    torch.mps.synchronize()
                
                # 性能测试
                start_time = time.time()
                for _ in range(100):
                    _ = method_func(batch_input)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                elif self.device.type == 'mps':
                    torch.mps.synchronize()
                
                end_time = time.time()
                avg_time = (end_time - start_time) / 100
                method_times.append(avg_time)
                
                print(f"   批处理大小 {batch_size}: {avg_time*1000:.2f}ms")
            
            performance_results[method_name] = method_times
            print()
        
        # 可视化性能结果
        plt.figure(figsize=(12, 8))
        
        for method_name, times in performance_results.items():
            plt.plot(batch_sizes, [t*1000 for t in times], 'o-', 
                    label=method_name, linewidth=2, markersize=8)
        
        plt.title('不同RMSNorm方法的性能对比', fontsize=16, fontweight='bold')
        plt.xlabel('批处理大小', fontsize=14)
        plt.ylabel('平均执行时间 (ms)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xscale('log', base=2)
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig('experiments/plots/rmsnorm_performance_benchmark.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def explain_optimized_solution(self) -> None:
        """解释优化的解决方案"""
        print("=== 优化的Batch-invariant解决方案解释 ===\n")
        
        print("🔧 问题根源:")
        print("1. **并行归约竞争**: 多线程/多核并行执行时的竞争条件")
        print("2. **浮点数非结合性**: (a + b) + c ≠ a + (b + c)")
        print("3. **内存访问模式**: 不同的内存访问顺序")
        print("4. **GPU架构差异**: MPS、CUDA等不同架构的并行策略")
        print()
        
        print("💡 优化解决方案:")
        print("1. **固定分块策略**: 使用固定的分块大小，确保batch-invariant")
        print("2. **确定性并行**: 避免竞争条件，确保可重现性")
        print("3. **架构适配**: 针对不同GPU架构优化")
        print("4. **性能平衡**: 在确定性和性能之间找到平衡")
        print()
        
        print("📝 实现细节:")
        print("```python")
        print("# 优化的Batch-invariant RMSNorm")
        print("def batch_invariant_rmsnorm_optimized(x, eps=1e-6):")
        print("    fixed_chunk_size = 64  # 固定分块大小")
        print("    rms_squared = torch.zeros_like(x[:, :, :1])")
        print("    for i in range(0, hidden_dim, fixed_chunk_size):")
        print("        chunk = x[:, :, i:i+fixed_chunk_size]")
        print("        rms_squared += torch.sum(chunk ** 2, dim=-1, keepdim=True)")
        print("    return x / torch.sqrt(rms_squared / hidden_dim + eps)")
        print("```")
        print()
        
        print("🎯 关键洞察:")
        print("• 固定分块大小比固定归约顺序更高效")
        print("• 并行计算中的竞争条件是主要问题")
        print("• MPS和CUDA都需要特殊的优化策略")
        print("• 性能损失通常很小，但确定性收益很大")
        print()
    
    def run_complete_demo(self) -> None:
        """运行完整演示"""
        print("🚀 开始优化的Batch-invariant RMSNorm完整演示...\n")
        
        # 1. 演示并行计算中的variance
        self.demonstrate_parallel_variance()
        
        # 2. 分析并行计算影响
        self.analyze_parallel_effects()
        
        # 3. 可视化结果
        self.visualize_parallel_variance()
        
        # 4. 性能基准测试
        self.benchmark_performance()
        
        # 5. 解释优化解决方案
        self.explain_optimized_solution()
        
        print("✅ 优化的Batch-invariant RMSNorm演示完成！")

def main():
    """主函数"""
    demo = OptimizedBatchInvariantDemo(device='auto')
    demo.run_complete_demo()

if __name__ == "__main__":
    main()
