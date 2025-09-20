#!/usr/bin/env python3
"""
Batch-invariant RMSNorm演示

根据Thinking Machines的blog，演示RMSNorm如何从batch-variant变成batch-invariant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import time

# 导入字体配置
try:
    from src.font_config import setup_chinese_fonts
except ImportError:
    from font_config import setup_chinese_fonts

setup_chinese_fonts()

class BatchInvariantRMSNormDemo:
    """Batch-invariant RMSNorm演示类"""
    
    def __init__(self):
        """初始化演示"""
        self.results = {}
        
    def create_test_data(self, batch_sizes: List[int], seq_len: int = 512, hidden_dim: int = 1024) -> Dict[int, torch.Tensor]:
        """创建不同批处理大小的测试数据"""
        data = {}
        torch.manual_seed(42)  # 固定种子确保可重现性
        
        for batch_size in batch_sizes:
            # 创建相同的输入数据，只是批处理大小不同
            # 每个样本都是相同的，这样我们可以观察批处理大小对结果的影响
            base_sample = torch.randn(seq_len, hidden_dim)
            batch_data = base_sample.unsqueeze(0).repeat(batch_size, 1, 1)
            data[batch_size] = batch_data
            
        return data
    
    def standard_rmsnorm(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """标准RMSNorm实现 - Batch-variant"""
        # 计算RMS (Root Mean Square)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
        # 归一化
        return x / rms
    
    def batch_variant_rmsnorm(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Batch-variant RMSNorm - 模拟GPU并行归约的非确定性"""
        batch_size, seq_len, hidden_dim = x.shape
        
        # 模拟GPU并行归约的非确定性
        # 在实际GPU实现中，归约的顺序可能因为并行执行而不同
        rms_squared = torch.zeros(batch_size, seq_len, 1, device=x.device)
        
        # 模拟不同的归约策略
        if batch_size == 1:
            # 单样本：顺序归约
            for i in range(hidden_dim):
                rms_squared += x[:, :, i:i+1] ** 2
        elif batch_size == 2:
            # 双样本：两两归约
            for i in range(0, hidden_dim, 2):
                if i + 1 < hidden_dim:
                    rms_squared += x[:, :, i:i+1] ** 2 + x[:, :, i+1:i+2] ** 2
                else:
                    rms_squared += x[:, :, i:i+1] ** 2
        elif batch_size == 4:
            # 四样本：四四归约
            for i in range(0, hidden_dim, 4):
                chunk_sum = torch.zeros_like(rms_squared)
                for j in range(min(4, hidden_dim - i)):
                    chunk_sum += x[:, :, i+j:i+j+1] ** 2
                rms_squared += chunk_sum
        else:
            # 其他情况：随机归约顺序
            indices = torch.randperm(hidden_dim)
            for i in indices:
                rms_squared += x[:, :, i:i+1] ** 2
        
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
    
    def demonstrate_batch_variance(self) -> None:
        """演示Batch-variance问题"""
        print("=== Batch-variance RMSNorm演示 ===\n")
        
        # 创建测试数据
        batch_sizes = [1, 2, 4, 8]
        seq_len, hidden_dim = 256, 512
        
        print(f"📊 测试参数:")
        print(f"   序列长度: {seq_len}")
        print(f"   隐藏维度: {hidden_dim}")
        print(f"   批处理大小: {batch_sizes}")
        print()
        
        # 创建相同的输入数据
        base_input = torch.randn(seq_len, hidden_dim)
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"🔧 测试批处理大小: {batch_size}")
            
            # 创建批处理数据（所有样本都相同）
            batch_input = base_input.unsqueeze(0).repeat(batch_size, 1, 1)
            
            # 标准RMSNorm
            std_output = self.standard_rmsnorm(batch_input)
            
            # Batch-variant RMSNorm
            variant_output = self.batch_variant_rmsnorm(batch_input)
            
            # Batch-invariant RMSNorm
            invariant_output = self.batch_invariant_rmsnorm(batch_input)
            
            # 计算差异
            std_variant_diff = torch.max(torch.abs(std_output - variant_output)).item()
            std_invariant_diff = torch.max(torch.abs(std_output - invariant_output)).item()
            variant_invariant_diff = torch.max(torch.abs(variant_output - invariant_output)).item()
            
            results[batch_size] = {
                'std_variant_diff': std_variant_diff,
                'std_invariant_diff': std_invariant_diff,
                'variant_invariant_diff': variant_invariant_diff,
                'std_output': std_output[0].detach().cpu().numpy(),  # 取第一个样本
                'variant_output': variant_output[0].detach().cpu().numpy(),
                'invariant_output': invariant_output[0].detach().cpu().numpy()
            }
            
            print(f"   标准 vs Batch-variant: {std_variant_diff:.2e}")
            print(f"   标准 vs Batch-invariant: {std_invariant_diff:.2e}")
            print(f"   Batch-variant vs Batch-invariant: {variant_invariant_diff:.2e}")
            print()
        
        self.results = results
        return results
    
    def analyze_batch_invariance(self) -> None:
        """分析Batch-invariance"""
        print("=== Batch-invariance分析 ===\n")
        
        if not self.results:
            print("请先运行 demonstrate_batch_variance()")
            return
        
        print("🔍 关键观察:")
        print("1. **Batch-variant RMSNorm**: 不同批处理大小产生不同结果")
        print("2. **Batch-invariant RMSNorm**: 相同输入产生相同结果，无论批处理大小")
        print("3. **差异来源**: 并行归约的顺序不同")
        print()
        
        # 分析第一个样本在不同批处理大小下的结果
        print("📈 第一个样本在不同批处理大小下的结果:")
        for batch_size, result in self.results.items():
            first_sample_std = result['std_output']
            first_sample_variant = result['variant_output']
            first_sample_invariant = result['invariant_output']
            
            print(f"   批处理大小 {batch_size}:")
            print(f"     标准输出前5个值: {first_sample_std[0, :5]}")
            print(f"     Variant输出前5个值: {first_sample_variant[0, :5]}")
            print(f"     Invariant输出前5个值: {first_sample_invariant[0, :5]}")
            print()
    
    def visualize_batch_variance(self) -> None:
        """可视化Batch-variance"""
        if not self.results:
            print("请先运行 demonstrate_batch_variance()")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 差异对比
        ax1 = axes[0, 0]
        batch_sizes = list(self.results.keys())
        std_variant_diffs = [self.results[bs]['std_variant_diff'] for bs in batch_sizes]
        std_invariant_diffs = [self.results[bs]['std_invariant_diff'] for bs in batch_sizes]
        variant_invariant_diffs = [self.results[bs]['variant_invariant_diff'] for bs in batch_sizes]
        
        x = np.arange(len(batch_sizes))
        width = 0.25
        
        ax1.bar(x - width, std_variant_diffs, width, label='标准 vs Variant', alpha=0.8)
        ax1.bar(x, std_invariant_diffs, width, label='标准 vs Invariant', alpha=0.8)
        ax1.bar(x + width, variant_invariant_diffs, width, label='Variant vs Invariant', alpha=0.8)
        
        ax1.set_xlabel('批处理大小', fontsize=12)
        ax1.set_ylabel('最大差异', fontsize=12)
        ax1.set_title('RMSNorm输出差异对比', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(batch_sizes)
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 输出分布对比
        ax2 = axes[0, 1]
        batch_size = 4  # 选择一个批处理大小进行详细分析
        result = self.results[batch_size]
        
        std_output = result['std_output'].flatten()
        variant_output = result['variant_output'].flatten()
        invariant_output = result['invariant_output'].flatten()
        
        ax2.hist(std_output, bins=50, alpha=0.5, label='标准RMSNorm', density=True)
        ax2.hist(variant_output, bins=50, alpha=0.5, label='Batch-variant', density=True)
        ax2.hist(invariant_output, bins=50, alpha=0.5, label='Batch-invariant', density=True)
        
        ax2.set_xlabel('输出值', fontsize=12)
        ax2.set_ylabel('密度', fontsize=12)
        ax2.set_title(f'批处理大小{batch_size}的输出分布', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 差异热图
        ax3 = axes[1, 0]
        batch_size = 4
        result = self.results[batch_size]
        
        # 计算差异矩阵
        std_variant_diff = np.abs(result['std_output'] - result['variant_output'])
        std_invariant_diff = np.abs(result['std_output'] - result['invariant_output'])
        
        # 显示前50x50的差异
        im1 = ax3.imshow(std_variant_diff[:50, :50], cmap='Reds', aspect='auto')
        ax3.set_title('标准 vs Batch-variant差异热图', fontsize=14, fontweight='bold')
        ax3.set_xlabel('隐藏维度', fontsize=12)
        ax3.set_ylabel('序列位置', fontsize=12)
        plt.colorbar(im1, ax=ax3)
        
        # 4. 累积差异
        ax4 = axes[1, 1]
        batch_sizes = list(self.results.keys())
        cumulative_diffs = []
        
        for bs in batch_sizes:
            result = self.results[bs]
            # 计算累积差异（所有元素的绝对差异之和）
            cumulative_diff = np.sum(np.abs(result['std_output'] - result['variant_output']))
            cumulative_diffs.append(cumulative_diff)
        
        ax4.plot(batch_sizes, cumulative_diffs, 'o-', linewidth=2, markersize=8)
        ax4.set_xlabel('批处理大小', fontsize=12)
        ax4.set_ylabel('累积差异', fontsize=12)
        ax4.set_title('累积差异随批处理大小变化', fontsize=14, fontweight='bold')
        ax4.set_xscale('log', base=2)
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('experiments/plots/batch_invariant_rmsnorm_demo.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def explain_solution(self) -> None:
        """解释解决方案"""
        print("=== Batch-invariant解决方案解释 ===\n")
        
        print("🔧 问题根源:")
        print("1. **并行归约顺序**: GPU并行执行时，归约的顺序可能不同")
        print("2. **浮点数非结合性**: (a + b) + c ≠ a + (b + c)")
        print("3. **批处理大小依赖**: 不同的批处理大小导致不同的并行策略")
        print()
        
        print("💡 解决方案:")
        print("1. **固定归约顺序**: 总是按相同的顺序进行归约")
        print("2. **Batch-invariant策略**: 确保相同输入产生相同输出")
        print("3. **确定性实现**: 避免依赖并行执行顺序")
        print()
        
        print("📝 实现细节:")
        print("```python")
        print("# Batch-variant (问题)")
        print("for i in range(hidden_dim):")
        print("    rms_squared += x[:, :, i:i+1] ** 2  # 顺序可能不同")
        print()
        print("# Batch-invariant (解决方案)")
        print("for i in range(hidden_dim):")
        print("    rms_squared += x[:, :, i:i+1] ** 2  # 固定顺序")
        print("```")
        print()
        
        print("🎯 关键洞察:")
        print("• 不是所有操作都需要batch-invariant")
        print("• 只有涉及归约的操作才需要")
        print("• 矩阵乘法本身是batch-invariant的")
        print("• 注意力机制中的归约需要特殊处理")
        print()
    
    def run_complete_demo(self) -> None:
        """运行完整演示"""
        print("🚀 开始Batch-invariant RMSNorm完整演示...\n")
        
        # 1. 演示Batch-variance问题
        self.demonstrate_batch_variance()
        
        # 2. 分析Batch-invariance
        self.analyze_batch_invariance()
        
        # 3. 可视化结果
        self.visualize_batch_variance()
        
        # 4. 解释解决方案
        self.explain_solution()
        
        print("✅ Batch-invariant RMSNorm演示完成！")

def main():
    """主函数"""
    demo = BatchInvariantRMSNormDemo()
    demo.run_complete_demo()

if __name__ == "__main__":
    main()
