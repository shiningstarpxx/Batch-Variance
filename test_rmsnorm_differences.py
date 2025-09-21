#!/usr/bin/env python3
"""
直接测试RMSNorm差异
验证Batch Variant和Batch Invariant版本的行为差异
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
    """Batch Variant RMSNorm - 模拟非确定性归约"""
    
    def __init__(self, dim: int, eps: float = 1e-6, num_splits: int = 4):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.num_splits = num_splits
    
    def forward(self, x):
        # 模拟非确定性归约：随机分割并随机合并
        if x.dim() == 4:  # (B, H, W, C)
            batch_size, height, width, dim = x.shape
            x_flat = x.reshape(-1, dim)  # (B*H*W, C)
        elif x.dim() == 3:  # (B, seq_len, dim)
            batch_size, seq_len, dim = x.shape
            x_flat = x.reshape(-1, dim)  # (B*seq_len, dim)
        else:  # (B, dim)
            batch_size, dim = x.shape
            x_flat = x
        
        # 随机分割策略：模拟非确定性的并行归约
        split_size = dim // self.num_splits
        rms_sums = []
        
        for i in range(self.num_splits):
            start_idx = i * split_size
            if i == self.num_splits - 1:  # 最后一个分割包含剩余元素
                end_idx = dim
            else:
                end_idx = (i + 1) * split_size
            
            # 计算每个分割的平方和
            split_x = x_flat[..., start_idx:end_idx]
            split_sum = torch.sum(split_x**2, dim=-1, keepdim=True)
            rms_sums.append(split_sum)
        
        # 模拟非确定性的合并顺序：随机打乱并累加
        random.shuffle(rms_sums)  # 随机打乱顺序
        
        # 使用累积求和来模拟非确定性的归约过程
        total_sum = rms_sums[0]
        for i in range(1, len(rms_sums)):
            total_sum = total_sum + rms_sums[i]  # 模拟浮点数非结合性
        
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

def test_rmsnorm_differences():
    """测试RMSNorm差异"""
    print("=" * 60)
    print("RMSNorm差异测试")
    print("=" * 60)
    
    device = get_device('auto')
    print(f"使用设备: {device}")
    
    # 创建测试数据
    torch.manual_seed(42)
    test_input = torch.randn(2, 512, device=device)  # (batch_size, dim)
    
    # 创建两个版本的RMSNorm
    variant_norm = BatchVariantRMSNorm(512, num_splits=4).to(device)
    invariant_norm = BatchInvariantRMSNorm(512, fixed_split_size=64).to(device)
    
    print(f"测试输入形状: {test_input.shape}")
    print(f"测试输入范围: [{test_input.min():.6f}, {test_input.max():.6f}]")
    
    # 测试Batch Variant版本的非确定性
    print("\n=== Batch Variant版本测试 ===")
    variant_outputs = []
    for trial in range(10):
        # 不设置种子，让随机性发挥作用
        with torch.no_grad():
            output = variant_norm(test_input)
            variant_outputs.append(output.cpu().numpy())
            print(f"  第{trial+1}次: 输出范围 [{output.min():.6f}, {output.max():.6f}]")
    
    # 检查Batch Variant的一致性
    variant_consistent = True
    reference = variant_outputs[0]
    for i, output in enumerate(variant_outputs[1:], 1):
        if not np.allclose(output, reference, atol=1e-6, rtol=1e-6):
            variant_consistent = False
            print(f"  ❌ 第{i+1}次输出与第1次不同")
        else:
            print(f"  ✅ 第{i+1}次输出与第1次相同")
    
    print(f"Batch Variant一致性: {'✅ 通过' if variant_consistent else '❌ 失败'}")
    
    # 测试Batch Invariant版本的确定性
    print("\n=== Batch Invariant版本测试 ===")
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
    invariant_consistent = True
    reference = invariant_outputs[0]
    for i, output in enumerate(invariant_outputs[1:], 1):
        if not np.allclose(output, reference, atol=1e-6, rtol=1e-6):
            invariant_consistent = False
            print(f"  ❌ 第{i+1}次输出与第1次不同")
        else:
            print(f"  ✅ 第{i+1}次输出与第1次相同")
    
    print(f"Batch Invariant一致性: {'✅ 通过' if invariant_consistent else '❌ 失败'}")
    
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
    print(f"Batch Variant一致性: {'✅ 通过' if variant_consistent else '❌ 失败'}")
    print(f"Batch Invariant一致性: {'✅ 通过' if invariant_consistent else '❌ 失败'}")
    print(f"两版本间差异: {difference:.2e}")
    
    if not variant_consistent and invariant_consistent:
        print("🎉 测试成功！Batch Variant产生非确定性，Batch Invariant保持确定性")
    elif variant_consistent and invariant_consistent:
        print("⚠️ 两个版本都保持一致性，可能需要调整随机性模拟")
    else:
        print("❌ 测试结果不符合预期")

if __name__ == "__main__":
    test_rmsnorm_differences()
