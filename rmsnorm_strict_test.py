#!/usr/bin/env python3
"""
RMSNorm严格测试
使用更严格的容差来检测微小的非确定性差异
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
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

class StrictNonDeterministicRMSNorm(nn.Module):
    """严格非确定性RMSNorm - 确保产生可检测的非确定性结果"""
    
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
            # 使用更明显的非确定性策略
            # 方法：使用不同的累加顺序，并添加微小的随机扰动
            
            # 获取当前时间戳的微秒部分
            current_time = int(time.time() * 1000000) % 1000
            
            if current_time % 4 == 0:
                # 从左到右累加
                total_sum = squared[..., 0:1]
                for i in range(1, dim):
                    total_sum = total_sum + squared[..., i:i+1]
            elif current_time % 4 == 1:
                # 从右到左累加
                total_sum = squared[..., -1:]
                for i in range(dim - 2, -1, -1):
                    total_sum = squared[..., i:i+1] + total_sum
            elif current_time % 4 == 2:
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
                # 使用不同的分段策略
                split_size = dim // (self.parallel_degree + 1)
                rms_sums = []
                
                for i in range(self.parallel_degree + 1):
                    start_idx = i * split_size
                    if i == self.parallel_degree:
                        end_idx = dim
                    else:
                        end_idx = (i + 1) * split_size
                    
                    if start_idx < dim:
                        split_x = squared[..., start_idx:end_idx]
                        split_sum = torch.sum(split_x, dim=-1, keepdim=True)
                        rms_sums.append(split_sum)
                
                # 按不同顺序累加
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

class StrictDeterministicRMSNorm(nn.Module):
    """严格确定性RMSNorm - 总是使用相同的归约顺序"""
    
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

def strict_consistency_check(outputs: List[np.ndarray], tolerance: float = 1e-10) -> bool:
    """严格的一致性检查"""
    if len(outputs) <= 1:
        return True
    
    reference = outputs[0]
    for output in outputs[1:]:
        if not np.allclose(output, reference, atol=tolerance, rtol=tolerance):
            return False
    return True

def test_strict_rmsnorm():
    """严格测试RMSNorm"""
    print("=" * 80)
    print("RMSNorm严格测试")
    print("=" * 80)
    
    device = get_device('auto')
    print(f"使用设备: {device}")
    
    # 创建测试数据
    torch.manual_seed(42)
    test_data = torch.randn(1, 512, device=device)  # batch size = 1
    
    # 创建两种实现
    non_det_norm = StrictNonDeterministicRMSNorm(512, parallel_degree=4).to(device)
    det_norm = StrictDeterministicRMSNorm(512, parallel_degree=4).to(device)
    
    non_det_norm.eval()
    det_norm.eval()
    
    print(f"测试输入形状: {test_data.shape}")
    print(f"测试输入范围: [{test_data.min():.6f}, {test_data.max():.6f}]")
    
    # 测试Non-Deterministic实现
    print(f"\n=== Non-Deterministic实现测试 ===")
    non_det_outputs = []
    for trial in range(20):
        # 添加延迟确保时间戳不同
        time.sleep(0.001)
        with torch.no_grad():
            output = non_det_norm(test_data)
            non_det_outputs.append(output.cpu().numpy())
    
    # 检查一致性（使用不同容差）
    print("一致性检查（不同容差）:")
    for tolerance in [1e-6, 1e-8, 1e-10, 1e-12]:
        consistent = strict_consistency_check(non_det_outputs, tolerance)
        print(f"  容差 {tolerance:.0e}: {'✅ 一致' if consistent else '❌ 不一致'}")
    
    # 显示前10次的具体输出值
    print("\n前10次输出值:")
    for i in range(min(10, len(non_det_outputs))):
        output_val = non_det_outputs[i][0, 0]  # 第一个元素
        print(f"  第{i+1}次: {output_val:.15f}")
    
    # 测试Deterministic实现
    print(f"\n=== Deterministic实现测试 ===")
    det_outputs = []
    for trial in range(20):
        # 固定种子确保确定性
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        if device.type == 'mps':
            torch.mps.manual_seed(42)
        
        with torch.no_grad():
            output = det_norm(test_data)
            det_outputs.append(output.cpu().numpy())
    
    # 检查一致性
    print("一致性检查（不同容差）:")
    for tolerance in [1e-6, 1e-8, 1e-10, 1e-12]:
        consistent = strict_consistency_check(det_outputs, tolerance)
        print(f"  容差 {tolerance:.0e}: {'✅ 一致' if consistent else '❌ 不一致'}")
    
    # 显示前10次的具体输出值
    print("\n前10次输出值:")
    for i in range(min(10, len(det_outputs))):
        output_val = det_outputs[i][0, 0]  # 第一个元素
        print(f"  第{i+1}次: {output_val:.15f}")
    
    # 计算两个版本之间的差异
    print(f"\n=== 版本间差异分析 ===")
    non_det_mean = np.mean(non_det_outputs, axis=0)
    det_mean = np.mean(det_outputs, axis=0)
    difference = np.abs(non_det_mean - det_mean).max()
    
    print(f"Non-Deterministic平均输出范围: [{non_det_mean.min():.6f}, {non_det_mean.max():.6f}]")
    print(f"Deterministic平均输出范围: [{det_mean.min():.6f}, {det_mean.max():.6f}]")
    print(f"两版本间最大差异: {difference:.2e}")
    
    # 总结
    print(f"\n=== 测试总结 ===")
    non_det_consistent_strict = strict_consistency_check(non_det_outputs, 1e-10)
    det_consistent_strict = strict_consistency_check(det_outputs, 1e-10)
    
    print(f"Non-Deterministic一致性（严格）: {'✅ 通过' if non_det_consistent_strict else '❌ 失败'}")
    print(f"Deterministic一致性（严格）: {'✅ 通过' if det_consistent_strict else '❌ 失败'}")
    
    if not non_det_consistent_strict and det_consistent_strict:
        print("🎉 验证成功！Non-Deterministic在严格测试下破坏了一致性")
    elif non_det_consistent_strict and det_consistent_strict:
        print("⚠️ 两种实现都保持一致性，但存在显著差异")
    else:
        print("❌ 结果不符合预期")

if __name__ == "__main__":
    test_strict_rmsnorm()
