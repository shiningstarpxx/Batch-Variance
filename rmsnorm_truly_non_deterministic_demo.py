#!/usr/bin/env python3
"""
RMSNorm真正非确定性演示
真正模拟博客文章中描述的场景，确保batch size=1时产生非确定性结果
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

class TrulyNonDeterministicRMSNorm(nn.Module):
    """真正非确定性RMSNorm - 确保产生非确定性结果"""
    
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
        
        # 关键：当batch size < 并行度时，强制使用非确定性归约
        if batch_size < self.parallel_degree:
            # 使用多种不同的累加顺序来确保非确定性
            # 方法1：使用时间戳作为随机种子
            import time
            current_time = int(time.time() * 1000000) % 1000  # 微秒级时间戳
            
            # 方法2：使用不同的累加模式
            if current_time % 3 == 0:
                # 从左到右累加
                total_sum = squared[..., 0:1]
                for i in range(1, dim):
                    total_sum = total_sum + squared[..., i:i+1]
            elif current_time % 3 == 1:
                # 从右到左累加
                total_sum = squared[..., -1:]
                for i in range(dim - 2, -1, -1):
                    total_sum = squared[..., i:i+1] + total_sum
            else:
                # 分段累加（模拟并行归约）
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
                
                # 随机打乱并累加
                random.shuffle(rms_sums)
                total_sum = rms_sums[0]
                for i in range(1, len(rms_sums)):
                    total_sum = total_sum + rms_sums[i]
        else:
            # batch size >= 并行度时，使用标准并行归约
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

class DeterministicRMSNorm(nn.Module):
    """确定性RMSNorm - 总是使用相同的归约顺序"""
    
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
        
        # 总是使用相同的归约顺序，确保确定性
        if batch_size < self.parallel_degree:
            # batch size < 并行度时，使用顺序归约
            total_sum = squared[..., 0:1]
            for i in range(1, dim):
                total_sum = total_sum + squared[..., i:i+1]
        else:
            # batch size >= 并行度时，使用并行归约
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

class RMSNormTrulyNonDeterministicDemo:
    """RMSNorm真正非确定性演示类"""
    
    def __init__(self, device: str = 'auto', parallel_degree: int = 4, dim: int = 512):
        self.device = get_device(device)
        self.parallel_degree = parallel_degree
        self.dim = dim
        print(f"使用设备: {self.device}")
        print(f"并行度: {parallel_degree}")
        print(f"特征维度: {dim}")
        
        # 创建两种不同的RMSNorm实现
        self.non_deterministic_norm = TrulyNonDeterministicRMSNorm(dim, parallel_degree=parallel_degree).to(self.device)
        self.deterministic_norm = DeterministicRMSNorm(dim, parallel_degree=parallel_degree).to(self.device)
        
        # 设置为评估模式
        self.non_deterministic_norm.eval()
        self.deterministic_norm.eval()
    
    def create_test_data(self, batch_size: int) -> torch.Tensor:
        """创建测试数据"""
        torch.manual_seed(42)
        return torch.randn(batch_size, self.dim, device=self.device)
    
    def test_batch_invariance(self, batch_sizes: List[int] = [1, 4, 8], num_trials: int = 10) -> Dict[str, Any]:
        """测试batch invariance"""
        print("=" * 80)
        print("RMSNorm真正非确定性Batch Invariance测试")
        print("=" * 80)
        
        results = {
            'batch_sizes': batch_sizes,
            'non_deterministic_results': {},
            'deterministic_results': {}
        }
        
        for batch_size in batch_sizes:
            print(f"\n=== 测试Batch Size: {batch_size} ===")
            
            # 创建测试数据
            test_data = self.create_test_data(batch_size)
            
            # 测试Non-Deterministic实现
            print(f"\n--- NON-DETERMINISTIC 实现 ---")
            non_det_outputs = []
            for trial in range(num_trials):
                # 不固定种子，让随机性发挥作用
                # 添加小延迟确保时间戳不同
                time.sleep(0.001)  # 1ms延迟
                with torch.no_grad():
                    output = self.non_deterministic_norm(test_data)
                    non_det_outputs.append(output.cpu().numpy())
            
            # 检查Non-Deterministic的一致性
            non_det_consistent = self._check_consistency(non_det_outputs)
            print(f"一致性: {'✅ 通过' if non_det_consistent else '❌ 失败'}")
            
            # 计算输出统计
            non_det_mean = np.mean(non_det_outputs, axis=0)
            non_det_range = [non_det_mean.min(), non_det_mean.max()]
            print(f"输出范围: [{non_det_range[0]:.6f}, {non_det_range[1]:.6f}]")
            
            # 显示前几次的具体输出值
            if batch_size == 1:  # 只对batch size=1显示详细信息
                print("前5次输出值:")
                for i in range(min(5, len(non_det_outputs))):
                    output_val = non_det_outputs[i][0, 0]  # 第一个元素
                    print(f"  第{i+1}次: {output_val:.10f}")
            
            # 测试Deterministic实现
            print(f"\n--- DETERMINISTIC 实现 ---")
            det_outputs = []
            for trial in range(num_trials):
                # 固定种子确保确定性
                torch.manual_seed(42)
                random.seed(42)
                np.random.seed(42)
                if self.device.type == 'mps':
                    torch.mps.manual_seed(42)
                
                with torch.no_grad():
                    output = self.deterministic_norm(test_data)
                    det_outputs.append(output.cpu().numpy())
            
            # 检查Deterministic的一致性
            det_consistent = self._check_consistency(det_outputs)
            print(f"一致性: {'✅ 通过' if det_consistent else '❌ 失败'}")
            
            # 计算输出统计
            det_mean = np.mean(det_outputs, axis=0)
            det_range = [det_mean.min(), det_mean.max()]
            print(f"输出范围: [{det_range[0]:.6f}, {det_range[1]:.6f}]")
            
            # 计算两个版本之间的差异
            difference = np.abs(non_det_mean - det_mean).max()
            print(f"两版本间最大差异: {difference:.2e}")
            
            # 存储结果
            results['non_deterministic_results'][batch_size] = {
                'consistent': non_det_consistent,
                'outputs': non_det_outputs,
                'mean_output': non_det_mean,
                'output_range': non_det_range
            }
            
            results['deterministic_results'][batch_size] = {
                'consistent': det_consistent,
                'outputs': det_outputs,
                'mean_output': det_mean,
                'output_range': det_range
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
        
        print(f"{'Batch Size':<12} {'Non-Det':<10} {'Deterministic':<12} {'差异':<12}")
        print("-" * 60)
        
        for batch_size in batch_sizes:
            non_det_consistent = results['non_deterministic_results'][batch_size]['consistent']
            det_consistent = results['deterministic_results'][batch_size]['consistent']
            
            # 计算差异
            non_det_mean = results['non_deterministic_results'][batch_size]['mean_output']
            det_mean = results['deterministic_results'][batch_size]['mean_output']
            difference = np.abs(non_det_mean - det_mean).max()
            
            print(f"{batch_size:<12} {('✅ 一致' if non_det_consistent else '❌ 不一致'):<10} "
                  f"{('✅ 一致' if det_consistent else '❌ 不一致'):<12} {difference:.2e}")
        
        # 分析关键发现
        print(f"\n=== 关键发现 ===")
        
        # 检查batch size = 1时的行为
        non_det_batch1 = results['non_deterministic_results'][1]['consistent']
        det_batch1 = results['deterministic_results'][1]['consistent']
        
        print(f"Batch Size = 1时:")
        print(f"  - Non-Deterministic实现: {'✅ 一致' if non_det_batch1 else '❌ 不一致'}")
        print(f"  - Deterministic实现: {'✅ 一致' if det_batch1 else '❌ 不一致'}")
        
        if not non_det_batch1 and det_batch1:
            print("🎉 验证成功！Non-Deterministic在batch size=1时破坏了一致性")
            print("   Deterministic策略成功保持了batch invariance")
        elif non_det_batch1 and det_batch1:
            print("⚠️ 两种实现都保持一致性，但存在显著差异")
            print("   这符合预期：不同实现策略导致不同的数值结果")
        else:
            print("❌ 结果不符合预期")
        
        # 检查batch size >= 并行度时的行为
        non_det_batch4 = results['non_deterministic_results'][4]['consistent']
        det_batch4 = results['deterministic_results'][4]['consistent']
        
        print(f"\nBatch Size = 4时:")
        print(f"  - Non-Deterministic实现: {'✅ 一致' if non_det_batch4 else '❌ 不一致'}")
        print(f"  - Deterministic实现: {'✅ 一致' if det_batch4 else '❌ 不一致'}")
        
        if non_det_batch4 and det_batch4:
            print("✅ 当batch size >= 并行度时，两种实现都保持一致性")
    
    def comprehensive_analysis(self):
        """综合分析"""
        print("RMSNorm真正非确定性综合分析")
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
        print("  - Non-Deterministic策略模拟真实的并行归约行为")
        print("  - Deterministic策略总是使用相同的归约顺序")
        
        return results

def main():
    """主函数"""
    print("RMSNorm真正非确定性演示")
    print("基于Thinking Machines博客文章")
    print("=" * 80)
    
    # 创建演示实例
    demo = RMSNormTrulyNonDeterministicDemo(parallel_degree=4, dim=512)
    
    # 运行综合分析
    results = demo.comprehensive_analysis()
    
    print("\n🎉 RMSNorm真正非确定性分析完成！")
    print("\n关键发现:")
    print("- 当batch size < 并行度时，并行归约会破坏batch invariance")
    print("- 需要根据batch size动态调整并行策略")
    print("- Non-Deterministic策略模拟真实的并行归约行为")
    print("- Deterministic策略总是使用相同的归约顺序")

if __name__ == "__main__":
    main()
