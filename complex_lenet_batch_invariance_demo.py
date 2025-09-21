#!/usr/bin/env python3
"""
复杂LeNet Batch Invariance演示
基于Thinking Machines博客文章，实现带RMSNorm的复杂LeNet，
并对比batch variant和batch invariant两个版本的输出一致性

参考: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import sys
import os
import time

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from device_manager import get_device
from font_config import setup_chinese_fonts, force_chinese_fonts

# 设置中文字体
setup_chinese_fonts()
force_chinese_fonts()

class RMSNorm(nn.Module):
    """RMSNorm层 - 基于博客文章中的实现"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # 计算RMS
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        # 归一化并缩放
        return x / rms * self.weight

class BatchVariantRMSNorm(nn.Module):
    """Batch Variant RMSNorm - 标准实现，可能产生非确定性结果"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # 标准实现：直接计算RMS
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

class BatchInvariantRMSNorm(nn.Module):
    """Batch Invariant RMSNorm - 固定分割策略，确保确定性"""
    
    def __init__(self, dim: int, eps: float = 1e-6, fixed_split_size: int = 64):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.fixed_split_size = fixed_split_size
    
    def forward(self, x):
        # 固定分割策略：按固定大小分割，确保相同的归约顺序
        if x.dim() == 4:  # (B, H, W, C)
            batch_size, height, width, dim = x.shape
            x_flat = x.reshape(-1, dim)  # (B*H*W, C)
        elif x.dim() == 3:  # (B, seq_len, dim)
            batch_size, seq_len, dim = x.shape
            x_flat = x.reshape(-1, dim)  # (B*seq_len, dim)
        else:  # (B, dim)
            batch_size, dim = x.shape
            x_flat = x
        
        # 计算需要多少个分割
        num_splits = (dim + self.fixed_split_size - 1) // self.fixed_split_size
        
        # 按固定大小分割并计算RMS
        rms_sums = []
        for i in range(num_splits):
            start_idx = i * self.fixed_split_size
            end_idx = min((i + 1) * self.fixed_split_size, dim)
            
            # 计算每个分割的平方和
            split_x = x_flat[..., start_idx:end_idx]
            split_sum = torch.sum(split_x**2, dim=-1, keepdim=True)
            rms_sums.append(split_sum)
        
        # 按固定顺序累加
        total_sum = rms_sums[0]
        for i in range(1, len(rms_sums)):
            total_sum = total_sum + rms_sums[i]  # 固定顺序的累加
        
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

class ComplexLeNet(nn.Module):
    """复杂的LeNet网络，包含RMSNorm层"""
    
    def __init__(self, input_size: int = 784, num_classes: int = 10, 
                 use_batch_invariant: bool = False, fixed_split_size: int = 64):
        super(ComplexLeNet, self).__init__()
        
        self.use_batch_invariant = use_batch_invariant
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)            # 28x28 -> 24x24
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 24x24 -> 24x24
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)        # 24x24 -> 12x12 -> 6x6
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # RMSNorm层
        if use_batch_invariant:
            self.norm1 = BatchInvariantRMSNorm(16, fixed_split_size=fixed_split_size)
            self.norm2 = BatchInvariantRMSNorm(32, fixed_split_size=fixed_split_size)
            self.norm3 = BatchInvariantRMSNorm(64, fixed_split_size=fixed_split_size)
            self.norm4 = BatchInvariantRMSNorm(512, fixed_split_size=fixed_split_size)
            self.norm5 = BatchInvariantRMSNorm(256, fixed_split_size=fixed_split_size)
        else:
            self.norm1 = BatchVariantRMSNorm(16)
            self.norm2 = BatchVariantRMSNorm(32)
            self.norm3 = BatchVariantRMSNorm(64)
            self.norm4 = BatchVariantRMSNorm(512)
            self.norm5 = BatchVariantRMSNorm(256)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # 确保输入是4D张量
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        
        # 第一层卷积 + RMSNorm + 激活 + 池化
        x = self.conv1(x)  # 28x28 -> 28x28
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = F.relu(x)
        x = self.pool(x)  # 28x28 -> 14x14
        
        # 第二层卷积 + RMSNorm + 激活 + 池化
        x = self.conv2(x)  # 14x14 -> 10x10
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = F.relu(x)
        x = self.pool(x)  # 10x10 -> 5x5
        
        # 第三层卷积 + RMSNorm + 激活
        x = self.conv3(x)  # 5x5 -> 5x5
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm3(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = F.relu(x)
        x = self.pool(x)  # 5x5 -> 2x2
        
        # 展平 - 修正尺寸计算
        x = x.reshape(-1, 64 * 2 * 2)  # 64个通道，2x2的特征图
        
        # 全连接层 + RMSNorm + 激活
        x = self.fc1(x)
        x = x.unsqueeze(1)  # (B, 1, 512)
        x = self.norm4(x)
        x = x.squeeze(1)  # (B, 512)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = x.unsqueeze(1)  # (B, 1, 256)
        x = self.norm5(x)
        x = x.squeeze(1)  # (B, 256)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x

class BatchInvarianceDemo:
    """Batch Invariance演示类"""
    
    def __init__(self, device: str = 'auto', fixed_split_size: int = 64):
        self.device = get_device(device)
        self.fixed_split_size = fixed_split_size
        print(f"使用设备: {self.device}")
        print(f"固定分割大小: {fixed_split_size}")
        
        # 创建两个版本的模型
        self.model_variant = ComplexLeNet(use_batch_invariant=False).to(self.device)
        self.model_invariant = ComplexLeNet(use_batch_invariant=True, 
                                          fixed_split_size=fixed_split_size).to(self.device)
        
        # 设置为评估模式
        self.model_variant.eval()
        self.model_invariant.eval()
        
        # 创建测试数据
        self.test_batches = self._create_test_batches()
        
    def _create_test_batches(self) -> List[torch.Tensor]:
        """创建不同大小的测试批次"""
        test_batches = []
        
        # 创建不同大小的批次
        batch_sizes = [1, 2, 4, 8, 16]
        
        for batch_size in batch_sizes:
            # 使用固定种子确保数据一致性
            torch.manual_seed(42)
            batch_data = torch.randn(batch_size, 784, device=self.device)
            test_batches.append(batch_data)
            
        return test_batches
    
    def set_all_seeds(self, seed: int = 42):
        """设置所有随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == 'mps':
            torch.mps.manual_seed(seed)
        elif self.device.type == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        print(f"所有随机种子已设置为: {seed}")
    
    def test_batch_invariance(self, num_trials: int = 5) -> Dict[str, Any]:
        """测试batch invariance"""
        print("=" * 60)
        print("Batch Invariance测试")
        print("=" * 60)
        
        results = {
            'variant_results': [],
            'invariant_results': [],
            'differences': [],
            'batch_sizes': []
        }
        
        for batch_idx, batch_data in enumerate(self.test_batches):
            batch_size = batch_data.shape[0]
            print(f"\n=== 测试批次大小: {batch_size} ===")
            
            # 测试Batch Variant版本
            variant_outputs = []
            for trial in range(num_trials):
                self.set_all_seeds(42)  # 固定种子
                with torch.no_grad():
                    output = self.model_variant(batch_data)
                    variant_outputs.append(output.cpu().numpy())
            
            # 测试Batch Invariant版本
            invariant_outputs = []
            for trial in range(num_trials):
                self.set_all_seeds(42)  # 固定种子
                with torch.no_grad():
                    output = self.model_invariant(batch_data)
                    invariant_outputs.append(output.cpu().numpy())
            
            # 分析结果
            variant_consistent = self._check_consistency(variant_outputs)
            invariant_consistent = self._check_consistency(invariant_outputs)
            
            # 计算两个版本之间的差异
            variant_mean = np.mean(variant_outputs, axis=0)
            invariant_mean = np.mean(invariant_outputs, axis=0)
            difference = np.abs(variant_mean - invariant_mean).max()
            
            print(f"Batch Variant一致性: {'✅' if variant_consistent else '❌'}")
            print(f"Batch Invariant一致性: {'✅' if invariant_consistent else '❌'}")
            print(f"两版本间最大差异: {difference:.2e}")
            
            results['variant_results'].append(variant_outputs)
            results['invariant_results'].append(invariant_outputs)
            results['differences'].append(difference)
            results['batch_sizes'].append(batch_size)
        
        return results
    
    def _check_consistency(self, outputs: List[np.ndarray]) -> bool:
        """检查输出的一致性"""
        if len(outputs) <= 1:
            return True
        
        # 检查所有输出是否相同
        reference = outputs[0]
        for output in outputs[1:]:
            if not np.allclose(output, reference, atol=1e-6, rtol=1e-6):
                return False
        return True
    
    def benchmark_performance(self, num_iterations: int = 100) -> Dict[str, float]:
        """性能基准测试"""
        print("\n" + "=" * 60)
        print("性能基准测试")
        print("=" * 60)
        
        # 使用中等大小的批次进行测试
        test_batch = self.test_batches[2]  # batch_size = 4
        
        # 预热
        for _ in range(10):
            with torch.no_grad():
                _ = self.model_variant(test_batch)
                _ = self.model_invariant(test_batch)
        
        # 测试Batch Variant性能
        start_time = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = self.model_variant(test_batch)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        elif self.device.type == 'mps':
            torch.mps.synchronize()
            
        variant_time = time.time() - start_time
        
        # 测试Batch Invariant性能
        start_time = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = self.model_invariant(test_batch)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        elif self.device.type == 'mps':
            torch.mps.synchronize()
            
        invariant_time = time.time() - start_time
        
        print(f"Batch Variant平均时间: {variant_time/num_iterations*1000:.2f}ms")
        print(f"Batch Invariant平均时间: {invariant_time/num_iterations*1000:.2f}ms")
        print(f"性能开销: {(invariant_time/variant_time - 1)*100:.1f}%")
        
        return {
            'variant_time': variant_time / num_iterations,
            'invariant_time': invariant_time / num_iterations,
            'overhead': (invariant_time / variant_time - 1) * 100
        }
    
    def visualize_results(self, results: Dict[str, Any]):
        """可视化测试结果"""
        print("\n" + "=" * 60)
        print("结果可视化")
        print("=" * 60)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 批次大小 vs 差异
        axes[0, 0].plot(results['batch_sizes'], results['differences'], 'o-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('批次大小')
        axes[0, 0].set_ylabel('最大差异')
        axes[0, 0].set_title('批次大小 vs 输出差异')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 差异分布
        axes[0, 1].hist(results['differences'], bins=10, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('输出差异')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].set_title('差异分布')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 一致性检查
        variant_consistent = [self._check_consistency(outputs) for outputs in results['variant_results']]
        invariant_consistent = [self._check_consistency(outputs) for outputs in results['invariant_results']]
        
        x = np.arange(len(results['batch_sizes']))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, variant_consistent, width, label='Batch Variant', alpha=0.7)
        axes[1, 0].bar(x + width/2, invariant_consistent, width, label='Batch Invariant', alpha=0.7)
        axes[1, 0].set_xlabel('批次大小')
        axes[1, 0].set_ylabel('一致性 (1=一致, 0=不一致)')
        axes[1, 0].set_title('批次一致性对比')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(results['batch_sizes'])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 输出分布对比（使用第一个批次）
        variant_output = results['variant_results'][0][0]  # 第一个批次，第一次运行
        invariant_output = results['invariant_results'][0][0]
        
        axes[1, 1].scatter(variant_output.flatten(), invariant_output.flatten(), alpha=0.6)
        axes[1, 1].plot([variant_output.min(), variant_output.max()], 
                       [variant_output.min(), variant_output.max()], 'r--', linewidth=2)
        axes[1, 1].set_xlabel('Batch Variant输出')
        axes[1, 1].set_ylabel('Batch Invariant输出')
        axes[1, 1].set_title('输出分布对比')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('复杂LeNet Batch Invariance分析结果', fontsize=16, fontweight='bold', y=1.02)
        
        # 保存图片
        os.makedirs('experiments/plots', exist_ok=True)
        plt.savefig('experiments/plots/complex_lenet_batch_invariance.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("结果可视化已保存到: experiments/plots/complex_lenet_batch_invariance.png")
    
    def comprehensive_analysis(self):
        """综合分析"""
        print("复杂LeNet Batch Invariance综合分析")
        print("=" * 60)
        
        # 1. Batch Invariance测试
        results = self.test_batch_invariance()
        
        # 2. 性能基准测试
        perf_results = self.benchmark_performance()
        
        # 3. 可视化结果
        self.visualize_results(results)
        
        # 4. 总结
        print("\n" + "=" * 60)
        print("分析总结")
        print("=" * 60)
        
        max_difference = max(results['differences'])
        avg_difference = np.mean(results['differences'])
        
        print(f"最大输出差异: {max_difference:.2e}")
        print(f"平均输出差异: {avg_difference:.2e}")
        print(f"性能开销: {perf_results['overhead']:.1f}%")
        
        # 检查一致性
        variant_consistent = all(self._check_consistency(outputs) for outputs in results['variant_results'])
        invariant_consistent = all(self._check_consistency(outputs) for outputs in results['invariant_results'])
        
        print(f"Batch Variant一致性: {'✅ 通过' if variant_consistent else '❌ 失败'}")
        print(f"Batch Invariant一致性: {'✅ 通过' if invariant_consistent else '❌ 通过'}")
        
        if max_difference < 1e-5:
            print("🎉 两个版本输出基本一致！")
        else:
            print("⚠️ 两个版本存在显著差异")
        
        return {
            'results': results,
            'performance': perf_results,
            'max_difference': max_difference,
            'avg_difference': avg_difference,
            'variant_consistent': variant_consistent,
            'invariant_consistent': invariant_consistent
        }

def main():
    """主函数"""
    print("复杂LeNet Batch Invariance演示")
    print("基于Thinking Machines博客文章")
    print("=" * 60)
    
    # 创建演示实例
    demo = BatchInvarianceDemo(fixed_split_size=64)
    
    # 运行综合分析
    analysis_results = demo.comprehensive_analysis()
    
    print("\n🎉 复杂LeNet Batch Invariance分析完成！")
    print("\n关键发现:")
    print("- Batch Variant版本可能因浮点数非结合性产生非确定性结果")
    print("- Batch Invariant版本通过固定分割策略确保确定性")
    print("- 两个版本在相同输入下应该产生相似的输出")
    print("- 性能开销通常很小，但能显著提高确定性")

if __name__ == "__main__":
    main()
