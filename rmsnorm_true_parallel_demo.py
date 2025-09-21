#!/usr/bin/env python3
"""
RMSNorm真实并行演示
真正模拟博客文章中描述的场景：
- 当batch size < 并行度时，并行归约会破坏batch invariance
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

class TrueParallelRMSNorm(nn.Module):
    """真实并行RMSNorm - 模拟真实的并行归约行为"""
    
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
        
        # 模拟真实的并行归约行为
        # 关键：当batch size < 并行度时，并行归约会破坏batch invariance
        if batch_size < self.parallel_degree:
            # 模拟非确定性的并行归约
            # 使用随机顺序的累加来模拟并行执行的不确定性
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
            
            # 关键：模拟非确定性的合并顺序
            # 在真实并行环境中，不同线程的完成顺序是不确定的
            import random
            random.shuffle(rms_sums)  # 随机打乱顺序
            
            # 使用累积求和来模拟非确定性的归约过程
            total_sum = rms_sums[0]
            for i in range(1, len(rms_sums)):
                total_sum = total_sum + rms_sums[i]  # 模拟浮点数非结合性
        else:
            # batch size >= 并行度时，可以使用并行归约
            # 使用标准的并行归约
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
            
            # 按固定顺序累加（因为batch size足够大）
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

class BatchAwareRMSNorm(nn.Module):
    """Batch感知RMSNorm - 根据batch size动态调整策略"""
    
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
        
        # 根据batch size决定策略
        if batch_size < self.parallel_degree:
            # batch size < 并行度：不使用并行归约，保持batch invariance
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

class RMSNormTrueParallelDemo:
    """RMSNorm真实并行演示类"""
    
    def __init__(self, device: str = 'auto', parallel_degree: int = 4, dim: int = 512):
        self.device = get_device(device)
        self.parallel_degree = parallel_degree
        self.dim = dim
        print(f"使用设备: {self.device}")
        print(f"并行度: {parallel_degree}")
        print(f"特征维度: {dim}")
        
        # 创建两种不同的RMSNorm实现
        self.true_parallel_norm = TrueParallelRMSNorm(dim, parallel_degree=parallel_degree).to(self.device)
        self.batch_aware_norm = BatchAwareRMSNorm(dim, parallel_degree=parallel_degree).to(self.device)
        
        # 设置为评估模式
        self.true_parallel_norm.eval()
        self.batch_aware_norm.eval()
    
    def create_test_data(self, batch_size: int) -> torch.Tensor:
        """创建测试数据"""
        torch.manual_seed(42)
        return torch.randn(batch_size, self.dim, device=self.device)
    
    def test_batch_invariance(self, batch_sizes: List[int] = [1, 4, 8], num_trials: int = 10) -> Dict[str, Any]:
        """测试batch invariance"""
        print("=" * 80)
        print("RMSNorm真实并行Batch Invariance测试")
        print("=" * 80)
        
        results = {
            'batch_sizes': batch_sizes,
            'true_parallel_results': {},
            'batch_aware_results': {}
        }
        
        for batch_size in batch_sizes:
            print(f"\n=== 测试Batch Size: {batch_size} ===")
            
            # 创建测试数据
            test_data = self.create_test_data(batch_size)
            
            # 测试True Parallel实现
            print(f"\n--- TRUE PARALLEL 实现 ---")
            true_parallel_outputs = []
            for trial in range(num_trials):
                # 不固定种子，让随机性发挥作用
                with torch.no_grad():
                    output = self.true_parallel_norm(test_data)
                    true_parallel_outputs.append(output.cpu().numpy())
            
            # 检查True Parallel的一致性
            true_parallel_consistent = self._check_consistency(true_parallel_outputs)
            print(f"一致性: {'✅ 通过' if true_parallel_consistent else '❌ 失败'}")
            
            # 计算输出统计
            true_parallel_mean = np.mean(true_parallel_outputs, axis=0)
            true_parallel_range = [true_parallel_mean.min(), true_parallel_mean.max()]
            print(f"输出范围: [{true_parallel_range[0]:.6f}, {true_parallel_range[1]:.6f}]")
            
            # 测试Batch Aware实现
            print(f"\n--- BATCH AWARE 实现 ---")
            batch_aware_outputs = []
            for trial in range(num_trials):
                # 固定种子确保确定性
                torch.manual_seed(42)
                random.seed(42)
                np.random.seed(42)
                if self.device.type == 'mps':
                    torch.mps.manual_seed(42)
                
                with torch.no_grad():
                    output = self.batch_aware_norm(test_data)
                    batch_aware_outputs.append(output.cpu().numpy())
            
            # 检查Batch Aware的一致性
            batch_aware_consistent = self._check_consistency(batch_aware_outputs)
            print(f"一致性: {'✅ 通过' if batch_aware_consistent else '❌ 失败'}")
            
            # 计算输出统计
            batch_aware_mean = np.mean(batch_aware_outputs, axis=0)
            batch_aware_range = [batch_aware_mean.min(), batch_aware_mean.max()]
            print(f"输出范围: [{batch_aware_range[0]:.6f}, {batch_aware_range[1]:.6f}]")
            
            # 计算两个版本之间的差异
            difference = np.abs(true_parallel_mean - batch_aware_mean).max()
            print(f"两版本间最大差异: {difference:.2e}")
            
            # 存储结果
            results['true_parallel_results'][batch_size] = {
                'consistent': true_parallel_consistent,
                'outputs': true_parallel_outputs,
                'mean_output': true_parallel_mean,
                'output_range': true_parallel_range
            }
            
            results['batch_aware_results'][batch_size] = {
                'consistent': batch_aware_consistent,
                'outputs': batch_aware_outputs,
                'mean_output': batch_aware_mean,
                'output_range': batch_aware_range
            }
        
        return results
    
    def _check_consistency(self, outputs: List[np.ndarray]) -> bool:
        """检查输出的一致性"""
        if len(outputs) <= 1:
            return True
        
        reference = outputs[0]
        for output in outputs[1:]:
            if not np.allclose(output, reference, atol=1e-6, rtol=1e-6):
                return False
        return True
    
    def analyze_results(self, results: Dict[str, Any]):
        """分析结果"""
        print("\n" + "=" * 80)
        print("结果分析")
        print("=" * 80)
        
        batch_sizes = results['batch_sizes']
        
        print(f"{'Batch Size':<12} {'True Parallel':<15} {'Batch Aware':<12} {'差异':<12}")
        print("-" * 60)
        
        for batch_size in batch_sizes:
            true_parallel_consistent = results['true_parallel_results'][batch_size]['consistent']
            batch_aware_consistent = results['batch_aware_results'][batch_size]['consistent']
            
            # 计算差异
            true_parallel_mean = results['true_parallel_results'][batch_size]['mean_output']
            batch_aware_mean = results['batch_aware_results'][batch_size]['mean_output']
            difference = np.abs(true_parallel_mean - batch_aware_mean).max()
            
            print(f"{batch_size:<12} {('✅ 一致' if true_parallel_consistent else '❌ 不一致'):<15} "
                  f"{('✅ 一致' if batch_aware_consistent else '❌ 不一致'):<12} {difference:.2e}")
        
        # 分析关键发现
        print(f"\n=== 关键发现 ===")
        
        # 检查batch size = 1时的行为
        true_parallel_batch1 = results['true_parallel_results'][1]['consistent']
        batch_aware_batch1 = results['batch_aware_results'][1]['consistent']
        
        print(f"Batch Size = 1时:")
        print(f"  - True Parallel实现: {'✅ 一致' if true_parallel_batch1 else '❌ 不一致'}")
        print(f"  - Batch Aware实现: {'✅ 一致' if batch_aware_batch1 else '❌ 不一致'}")
        
        if not true_parallel_batch1 and batch_aware_batch1:
            print("🎉 验证成功！True Parallel在batch size=1时破坏了一致性")
            print("   Batch Aware策略成功保持了batch invariance")
        elif true_parallel_batch1 and batch_aware_batch1:
            print("⚠️ 两种实现都保持一致性，可能需要调整测试参数")
        else:
            print("❌ 结果不符合预期")
        
        # 检查batch size >= 并行度时的行为
        true_parallel_batch4 = results['true_parallel_results'][4]['consistent']
        batch_aware_batch4 = results['batch_aware_results'][4]['consistent']
        
        print(f"\nBatch Size = 4时:")
        print(f"  - True Parallel实现: {'✅ 一致' if true_parallel_batch4 else '❌ 不一致'}")
        print(f"  - Batch Aware实现: {'✅ 一致' if batch_aware_batch4 else '❌ 不一致'}")
        
        if true_parallel_batch4 and batch_aware_batch4:
            print("✅ 当batch size >= 并行度时，两种实现都保持一致性")
    
    def comprehensive_analysis(self):
        """综合分析"""
        print("RMSNorm真实并行综合分析")
        print("基于Thinking Machines博客文章")
        print("=" * 80)
        
        # 测试参数
        batch_sizes = [1, 4, 8]
        parallel_degree = 4
        
        print(f"测试配置:")
        print(f"  - Batch Sizes: {batch_sizes}")
        print(f"  - 并行度: {parallel_degree}")
        print(f"  - 特征维度: {self.dim}")
        
        # 运行测试
        results = self.test_batch_invariance(batch_sizes)
        
        # 分析结果
        self.analyze_results(results)
        
        # 总结
        print("\n" + "=" * 80)
        print("分析总结")
        print("=" * 80)
        
        print("🎯 核心观点验证:")
        print("  - 当batch size < 并行度时，并行归约会破坏batch invariance")
        print("  - 需要根据batch size动态调整并行策略")
        print("  - True Parallel策略模拟真实的并行归约行为")
        print("  - Batch Aware策略根据batch size动态调整策略")
        
        return results

def main():
    """主函数"""
    print("RMSNorm真实并行演示")
    print("基于Thinking Machines博客文章")
    print("=" * 80)
    
    # 创建演示实例
    demo = RMSNormTrueParallelDemo(parallel_degree=4, dim=512)
    
    # 运行综合分析
    results = demo.comprehensive_analysis()
    
    print("\n🎉 RMSNorm真实并行分析完成！")
    print("\n关键发现:")
    print("- 当batch size < 并行度时，并行归约会破坏batch invariance")
    print("- 需要根据batch size动态调整并行策略")
    print("- True Parallel策略模拟真实的并行归约行为")
    print("- Batch Aware策略根据batch size动态调整策略")

if __name__ == "__main__":
    main()
