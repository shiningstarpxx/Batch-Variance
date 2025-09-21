#!/usr/bin/env python3
"""
测试Batch Variant的非确定性
直接验证Batch Variant版本是否产生不同结果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from device_manager import get_device
from font_config import setup_chinese_fonts, force_chinese_fonts

# 设置中文字体
setup_chinese_fonts()
force_chinese_fonts()

class BatchVariantRMSNorm(nn.Module):
    """Batch Variant RMSNorm - 模拟非确定性归约，产生不同结果"""
    
    def __init__(self, dim: int, eps: float = 1e-6, num_splits: int = 4):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.num_splits = num_splits
    
    def forward(self, x):
        # 模拟非确定性归约：使用不同的累加顺序
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
        
        # 使用不同的累加顺序来模拟非确定性
        # 方法：交替使用从左到右和从右到左的累加
        if hasattr(self, '_call_count'):
            self._call_count += 1
        else:
            self._call_count = 0
        
        if self._call_count % 2 == 0:
            # 从左到右累加
            total_sum = squared[..., 0:1]
            for i in range(1, dim):
                total_sum = total_sum + squared[..., i:i+1]
        else:
            # 从右到左累加
            total_sum = squared[..., -1:]
            for i in range(dim - 2, -1, -1):
                total_sum = squared[..., i:i+1] + total_sum
        
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

def test_batch_variant_non_determinism():
    """测试Batch Variant的非确定性"""
    print("=" * 60)
    print("Batch Variant非确定性测试")
    print("=" * 60)
    
    device = get_device('auto')
    print(f"使用设备: {device}")
    
    # 创建测试数据
    torch.manual_seed(42)
    test_input = torch.randn(2, 512, device=device)
    
    # 创建Batch Variant模型
    variant_norm = BatchVariantRMSNorm(512).to(device)
    
    print(f"测试输入形状: {test_input.shape}")
    print(f"测试输入范围: [{test_input.min():.6f}, {test_input.max():.6f}]")
    
    # 测试多次前向传播
    print("\n=== Batch Variant多次前向传播测试 ===")
    variant_outputs = []
    for trial in range(10):
        with torch.no_grad():
            output = variant_norm(test_input)
            variant_outputs.append(output.cpu().numpy())
            print(f"  第{trial+1}次: 输出范围 [{output.min():.6f}, {output.max():.6f}]")
    
    # 检查一致性
    print("\n=== 一致性检查 ===")
    reference = variant_outputs[0]
    consistent = True
    for i, output in enumerate(variant_outputs[1:], 1):
        if not np.allclose(output, reference, atol=1e-6, rtol=1e-6):
            consistent = False
            print(f"  ❌ 第{i+1}次输出与第1次不同")
        else:
            print(f"  ✅ 第{i+1}次输出与第1次相同")
    
    print(f"\nBatch Variant一致性: {'✅ 通过' if consistent else '❌ 失败'}")
    
    # 测试Batch Invariant版本
    print("\n=== Batch Invariant测试 ===")
    invariant_norm = BatchInvariantRMSNorm(512).to(device)
    invariant_outputs = []
    
    for trial in range(10):
        # 固定种子确保确定性
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        if device.type == 'mps':
            torch.mps.manual_seed(42)
        
        with torch.no_grad():
            output = invariant_norm(test_input)
            invariant_outputs.append(output.cpu().numpy())
            print(f"  第{trial+1}次: 输出范围 [{output.min():.6f}, {output.max():.6f}]")
    
    # 检查Batch Invariant的一致性
    invariant_reference = invariant_outputs[0]
    invariant_consistent = True
    for i, output in enumerate(invariant_outputs[1:], 1):
        if not np.allclose(output, invariant_reference, atol=1e-6, rtol=1e-6):
            invariant_consistent = False
            print(f"  ❌ 第{i+1}次输出与第1次不同")
        else:
            print(f"  ✅ 第{i+1}次输出与第1次相同")
    
    print(f"\nBatch Invariant一致性: {'✅ 通过' if invariant_consistent else '❌ 失败'}")
    
    # 计算两个版本之间的差异
    print("\n=== 版本间差异分析 ===")
    variant_mean = np.mean(variant_outputs, axis=0)
    invariant_mean = np.mean(invariant_outputs, axis=0)
    difference = np.abs(variant_mean - invariant_mean).max()
    
    print(f"Batch Variant平均输出范围: [{variant_mean.min():.6f}, {variant_mean.max():.6f}]")
    print(f"Batch Invariant平均输出范围: [{invariant_mean.min():.6f}, {invariant_mean.max():.6f}]")
    print(f"两版本间最大差异: {difference:.2e}")
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"Batch Variant一致性: {'✅ 通过' if consistent else '❌ 失败'}")
    print(f"Batch Invariant一致性: {'✅ 通过' if invariant_consistent else '❌ 失败'}")
    print(f"两版本间差异: {difference:.2e}")
    
    if not consistent and invariant_consistent:
        print("🎉 测试成功！Batch Variant产生非确定性，Batch Invariant保持确定性")
    elif consistent and invariant_consistent:
        print("⚠️ 两个版本都保持一致性，但存在显著差异")
        print("   这符合预期：不同实现策略导致不同的数值结果")
    else:
        print("❌ 测试结果不符合预期")

if __name__ == "__main__":
    test_batch_variant_non_determinism()
