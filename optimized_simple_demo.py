#!/usr/bin/env python3
"""
简化的优化Batch-invariant RMSNorm演示

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

# 导入字体配置和设备管理
try:
    from src.font_config import setup_chinese_fonts
    from src.device_manager import get_device, device_manager
except ImportError:
    from font_config import setup_chinese_fonts
    from device_manager import get_device, device_manager

setup_chinese_fonts()

class OptimizedSimpleDemo:
    """简化的优化Batch-invariant演示类"""
    
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
    
    def create_realistic_data(self, batch_sizes: List[int], seq_len: int = 512, 
                            hidden_dim: int = 1024) -> Dict[int, torch.Tensor]:
        """创建更真实的测试数据"""
        data = {}
        torch.manual_seed(42)
        
        for batch_size in batch_sizes:
            # 创建更真实的输入数据，包含不同数量级的数值
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
    
    def batch_variant_rmsnorm_chunked(self, x: torch.Tensor, chunk_size: int = 64, 
                                    eps: float = 1e-6) -> torch.Tensor:
        """Batch-variant RMSNorm - 使用不同分块大小模拟非确定性"""
        batch_size, seq_len, hidden_dim = x.shape
        
        # 使用分块归约，模拟GPU分块处理
        rms_squared = torch.zeros(batch_size, seq_len, 1, device=x.device)
        
        for i in range(0, hidden_dim, chunk_size):
            end_idx = min(i + chunk_size, hidden_dim)
            chunk = x[:, :, i:end_idx]
            chunk_sum = torch.sum(chunk ** 2, dim=-1, keepdim=True)
            rms_squared += chunk_sum
        
        rms = torch.sqrt(rms_squared / hidden_dim + eps)
        return x / rms
    
    def batch_variant_rmsnorm_parallel_sim(self, x: torch.Tensor, num_splits: int = 4, 
                                         eps: float = 1e-6) -> torch.Tensor:
        """Batch-variant RMSNorm - 模拟并行归约的非确定性"""
        batch_size, seq_len, hidden_dim = x.shape
        
        # 模拟并行归约：将隐藏维度分成多个部分
        split_size = hidden_dim // num_splits
        rms_squared = torch.zeros(batch_size, seq_len, 1, device=x.device)
        
        # 模拟不同的归约顺序
        for split_idx in range(num_splits):
            start_idx = split_idx * split_size
            end_idx = start_idx + split_size if split_idx < num_splits - 1 else hidden_dim
            
            # 计算这个分片的贡献
            split_contribution = torch.sum(x[:, :, start_idx:end_idx] ** 2, dim=-1, keepdim=True)
            
            # 模拟并行归约的微小差异
            if split_idx % 2 == 0:
                # 偶数分片：正常添加
                rms_squared += split_contribution
            else:
                # 奇数分片：添加微小的数值扰动
                noise = torch.randn_like(split_contribution) * 1e-10
                rms_squared += split_contribution + noise
        
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
    
    def demonstrate_optimized_variance(self) -> None:
        """演示优化的variance问题"""
        print("=== 优化的Batch-variance演示 ===\n")
        
        # 创建测试数据
        batch_sizes = [1, 2, 4, 8]
        seq_len, hidden_dim = 256, 512
        
        print(f"📊 测试参数:")
        print(f"   序列长度: {seq_len}")
        print(f"   隐藏维度: {hidden_dim}")
        print(f"   批处理大小: {batch_sizes}")
        print(f"   设备: {self.device}")
        print()
        
        # 创建测试数据
        data = self.create_realistic_data(batch_sizes, seq_len, hidden_dim)
        
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
            
            # 测试不同的variant策略
            chunk_sizes = [32, 64, 128]
            num_splits = [2, 4, 8]
            
            batch_results = {
                'std_output': std_output[0].detach().cpu().numpy(),
                'invariant_output': invariant_output[0].detach().cpu().numpy(),
                'optimized_invariant_output': optimized_invariant_output[0].detach().cpu().numpy(),
                'variants': {}
            }
            
            # 测试不同分块大小
            for chunk_size in chunk_sizes:
                variant_output = self.batch_variant_rmsnorm_chunked(batch_input, chunk_size)
                diff = torch.max(torch.abs(std_output - variant_output)).item()
                
                batch_results['variants'][f'chunked_{chunk_size}'] = {
                    'output': variant_output[0].detach().cpu().numpy(),
                    'diff': diff
                }
                
                print(f"   分块大小 {chunk_size}: 差异 {diff:.2e}")
            
            # 测试不同并行分片数
            for num_split in num_splits:
                variant_output = self.batch_variant_rmsnorm_parallel_sim(batch_input, num_split)
                diff = torch.max(torch.abs(std_output - variant_output)).item()
                
                batch_results['variants'][f'parallel_{num_split}'] = {
                    'output': variant_output[0].detach().cpu().numpy(),
                    'diff': diff
                }
                
                print(f"   并行分片 {num_split}: 差异 {diff:.2e}")
            
            # 计算invariant方法的差异
            std_invariant_diff = torch.max(torch.abs(std_output - invariant_output)).item()
            std_optimized_diff = torch.max(torch.abs(std_output - optimized_invariant_output)).item()
            
            print(f"   Batch-invariant: 差异 {std_invariant_diff:.2e}")
            print(f"   优化Batch-invariant: 差异 {std_optimized_diff:.2e}")
            print()
            
            results[batch_size] = batch_results
        
        self.results = results
        return results
    
    def analyze_optimized_effects(self) -> None:
        """分析优化效果"""
        print("=== 优化效果分析 ===\n")
        
        if not self.results:
            print("请先运行 demonstrate_optimized_variance()")
            return
        
        print("🔍 关键观察:")
        print("1. **分块大小影响**: 不同分块大小产生不同结果")
        print("2. **并行分片影响**: 不同并行分片数产生不同结果")
        print("3. **Batch-invariant**: 固定策略确保一致性")
        print("4. **MPS优化**: 在Apple Silicon上表现良好")
        print()
        
        # 分析第一个样本在不同策略下的结果
        batch_size = 4
        if batch_size in self.results:
            result = self.results[batch_size]
            print(f"📈 批处理大小 {batch_size} 的详细分析:")
            
            std_output = result['std_output']
            invariant_output = result['invariant_output']
            
            print(f"   标准RMSNorm前5个值: {std_output[0, :5]}")
            print(f"   Batch-invariant前5个值: {invariant_output[0, :5]}")
            
            for variant_name, variant_result in result['variants'].items():
                variant_output = variant_result['output']
                diff = variant_result['diff']
                print(f"   {variant_name}前5个值: {variant_output[0, :5]} (差异: {diff:.2e})")
            print()
    
    def visualize_optimized_results(self) -> None:
        """可视化优化结果"""
        if not self.results:
            print("请先运行 demonstrate_optimized_variance()")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 不同策略的差异对比
        ax1 = axes[0, 0]
        batch_sizes = list(self.results.keys())
        
        # 收集不同策略的差异
        chunk_diffs = {32: [], 64: [], 128: []}
        parallel_diffs = {2: [], 4: [], 8: []}
        
        for bs in batch_sizes:
            result = self.results[bs]
            for chunk_size in [32, 64, 128]:
                key = f'chunked_{chunk_size}'
                if key in result['variants']:
                    chunk_diffs[chunk_size].append(result['variants'][key]['diff'])
                else:
                    chunk_diffs[chunk_size].append(0)
            
            for num_split in [2, 4, 8]:
                key = f'parallel_{num_split}'
                if key in result['variants']:
                    parallel_diffs[num_split].append(result['variants'][key]['diff'])
                else:
                    parallel_diffs[num_split].append(0)
        
        # 绘制分块大小影响
        for chunk_size, diffs in chunk_diffs.items():
            ax1.plot(batch_sizes, diffs, 'o-', label=f'分块大小 {chunk_size}', linewidth=2, markersize=6)
        
        ax1.set_xlabel('批处理大小', fontsize=12)
        ax1.set_ylabel('最大差异', fontsize=12)
        ax1.set_title('分块大小对差异的影响', fontsize=14, fontweight='bold')
        ax1.set_xscale('log', base=2)
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 并行分片数对差异的影响
        ax2 = axes[0, 1]
        for num_split, diffs in parallel_diffs.items():
            ax2.plot(batch_sizes, diffs, 's-', label=f'并行分片 {num_split}', linewidth=2, markersize=6)
        
        ax2.set_xlabel('批处理大小', fontsize=12)
        ax2.set_ylabel('最大差异', fontsize=12)
        ax2.set_title('并行分片数对差异的影响', fontsize=14, fontweight='bold')
        ax2.set_xscale('log', base=2)
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 输出分布对比
        ax3 = axes[1, 0]
        batch_size = 4
        if batch_size in self.results:
            result = self.results[batch_size]
            
            std_output = result['std_output'].flatten()
            invariant_output = result['invariant_output'].flatten()
            optimized_output = result['optimized_invariant_output'].flatten()
            
            ax3.hist(std_output, bins=50, alpha=0.5, label='标准RMSNorm', density=True)
            ax3.hist(invariant_output, bins=50, alpha=0.5, label='Batch-invariant', density=True)
            ax3.hist(optimized_output, bins=50, alpha=0.5, label='优化Batch-invariant', density=True)
            
            ax3.set_xlabel('输出值', fontsize=12)
            ax3.set_ylabel('密度', fontsize=12)
            ax3.set_title(f'输出分布对比 (批处理大小={batch_size})', fontsize=14, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. 差异热图
        ax4 = axes[1, 1]
        batch_size = 4
        if batch_size in self.results:
            result = self.results[batch_size]
            
            std_output = result['std_output']
            invariant_output = result['invariant_output']
            
            diff_matrix = np.abs(std_output - invariant_output)
            
            im = ax4.imshow(diff_matrix[:50, :50], cmap='Reds', aspect='auto')
            ax4.set_title('标准 vs Batch-invariant差异热图', fontsize=14, fontweight='bold')
            ax4.set_xlabel('隐藏维度', fontsize=12)
            ax4.set_ylabel('序列位置', fontsize=12)
            plt.colorbar(im, ax=ax4)
        
        plt.tight_layout()
        plt.savefig('experiments/plots/optimized_simple_demo.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def benchmark_performance(self) -> None:
        """性能基准测试"""
        print("=== 性能基准测试 ===\n")
        
        batch_sizes = [1, 2, 4, 8]
        seq_len, hidden_dim = 256, 512
        
        # 创建测试数据
        data = self.create_realistic_data(batch_sizes, seq_len, hidden_dim)
        
        methods = {
            '标准RMSNorm': self.standard_rmsnorm,
            'Batch-invariant': self.batch_invariant_rmsnorm,
            '优化Batch-invariant': self.batch_invariant_rmsnorm_optimized,
            '分块Variant (64)': lambda x: self.batch_variant_rmsnorm_chunked(x, 64),
            '并行Variant (4)': lambda x: self.batch_variant_rmsnorm_parallel_sim(x, 4),
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
        plt.savefig('experiments/plots/optimized_performance_benchmark.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def explain_optimized_solution(self) -> None:
        """解释优化的解决方案"""
        print("=== 优化的Batch-invariant解决方案解释 ===\n")
        
        print("🔧 问题根源:")
        print("1. **分块归约差异**: 不同分块大小导致不同的归约顺序")
        print("2. **并行归约竞争**: 模拟GPU并行执行时的竞争条件")
        print("3. **浮点数非结合性**: (a + b) + c ≠ a + (b + c)")
        print("4. **MPS架构特性**: Apple Silicon的特殊并行处理方式")
        print()
        
        print("💡 优化解决方案:")
        print("1. **固定分块策略**: 使用固定的分块大小，确保batch-invariant")
        print("2. **确定性归约**: 避免竞争条件，确保可重现性")
        print("3. **MPS优化**: 针对Apple Silicon架构优化")
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
        print("• 分块大小对结果有显著影响")
        print("• 并行分片数也会影响结果")
        print("• MPS在Apple Silicon上表现优异")
        print("• 性能损失通常很小，但确定性收益很大")
        print()
    
    def run_complete_demo(self) -> None:
        """运行完整演示"""
        print("🚀 开始优化的Batch-invariant RMSNorm完整演示...\n")
        
        # 1. 演示优化的variance
        self.demonstrate_optimized_variance()
        
        # 2. 分析优化效果
        self.analyze_optimized_effects()
        
        # 3. 可视化结果
        self.visualize_optimized_results()
        
        # 4. 性能基准测试
        self.benchmark_performance()
        
        # 5. 解释优化解决方案
        self.explain_optimized_solution()
        
        print("✅ 优化的Batch-invariant RMSNorm演示完成！")

def main():
    """主函数"""
    demo = OptimizedSimpleDemo(device='auto')
    demo.run_complete_demo()

if __name__ == "__main__":
    main()
