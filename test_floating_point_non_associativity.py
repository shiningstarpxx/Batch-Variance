#!/usr/bin/env python3
"""
测试浮点数非结合性
直接演示 (a + b) + c ≠ a + (b + c) 的影响
"""

import torch
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

def test_floating_point_non_associativity():
    """测试浮点数非结合性"""
    print("=" * 60)
    print("浮点数非结合性测试")
    print("=" * 60)
    
    device = get_device('auto')
    print(f"使用设备: {device}")
    
    # 创建一些会产生非结合性问题的数值
    # 使用不同数量级的数值来放大浮点数误差
    values = [
        1e10, 1e-10, 1e10, 1e-10, 1e10, 1e-10, 1e10, 1e-10,
        1e10, 1e-10, 1e10, 1e-10, 1e10, 1e-10, 1e10, 1e-10
    ]
    
    print(f"测试数值: {values[:8]}... (共{len(values)}个)")
    
    # 转换为tensor
    values_tensor = torch.tensor(values, device=device, dtype=torch.float32)
    
    # 方法1: 从左到右累加 (a + b) + c + d + ...
    left_to_right = values_tensor[0]
    for i in range(1, len(values_tensor)):
        left_to_right = left_to_right + values_tensor[i]
    
    # 方法2: 从右到左累加 ... + d + c + (b + a)
    right_to_left = values_tensor[-1]
    for i in range(len(values_tensor) - 2, -1, -1):
        right_to_left = values_tensor[i] + right_to_left
    
    # 方法3: 随机顺序累加
    random_order = values_tensor.clone()
    random.shuffle(random_order)
    random_sum = random_order[0]
    for i in range(1, len(random_order)):
        random_sum = random_sum + random_order[i]
    
    # 方法4: 使用PyTorch的sum函数
    pytorch_sum = torch.sum(values_tensor)
    
    print(f"\n=== 累加结果对比 ===")
    print(f"从左到右累加: {left_to_right.item():.15f}")
    print(f"从右到左累加: {right_to_left.item():.15f}")
    print(f"随机顺序累加: {random_sum.item():.15f}")
    print(f"PyTorch sum:   {pytorch_sum.item():.15f}")
    
    # 计算差异
    diff_lr_rl = abs(left_to_right - right_to_left).item()
    diff_lr_random = abs(left_to_right - random_sum).item()
    diff_rl_random = abs(right_to_left - random_sum).item()
    diff_lr_pytorch = abs(left_to_right - pytorch_sum).item()
    
    print(f"\n=== 差异分析 ===")
    print(f"左到右 vs 右到左: {diff_lr_rl:.2e}")
    print(f"左到右 vs 随机:   {diff_lr_random:.2e}")
    print(f"右到左 vs 随机:   {diff_rl_random:.2e}")
    print(f"左到右 vs PyTorch: {diff_lr_pytorch:.2e}")
    
    # 测试多次随机累加
    print(f"\n=== 多次随机累加测试 ===")
    random_results = []
    for trial in range(10):
        random_order = values_tensor.clone()
        random.shuffle(random_order)
        random_sum = random_order[0]
        for i in range(1, len(random_order)):
            random_sum = random_sum + random_order[i]
        random_results.append(random_sum.item())
        print(f"  第{trial+1}次随机累加: {random_sum.item():.15f}")
    
    # 分析随机结果的一致性
    unique_results = len(set([f"{r:.10f}" for r in random_results]))
    print(f"\n唯一结果数量: {unique_results}/10")
    
    if unique_results > 1:
        print("🎉 成功！随机累加产生了不同的结果，证明了浮点数非结合性")
    else:
        print("⚠️ 所有随机累加结果相同，可能需要调整数值或方法")

def test_rmsnorm_with_different_orders():
    """测试不同归约顺序对RMSNorm的影响"""
    print("\n" + "=" * 60)
    print("RMSNorm归约顺序测试")
    print("=" * 60)
    
    device = get_device('auto')
    
    # 创建测试数据
    torch.manual_seed(42)
    test_input = torch.randn(2, 512, device=device)
    
    print(f"测试输入形状: {test_input.shape}")
    print(f"测试输入范围: [{test_input.min():.6f}, {test_input.max():.6f}]")
    
    # 计算平方
    squared = test_input ** 2
    
    # 方法1: 直接使用torch.mean
    rms1 = torch.sqrt(torch.mean(squared, dim=-1, keepdim=True) + 1e-6)
    result1 = test_input / rms1
    
    # 方法2: 手动累加（从左到右）
    manual_sum = squared[..., 0:1]
    for i in range(1, squared.shape[-1]):
        manual_sum = manual_sum + squared[..., i:i+1]
    rms2 = torch.sqrt(manual_sum / squared.shape[-1] + 1e-6)
    result2 = test_input / rms2
    
    # 方法3: 手动累加（从右到左）
    manual_sum_rl = squared[..., -1:]
    for i in range(squared.shape[-1] - 2, -1, -1):
        manual_sum_rl = squared[..., i:i+1] + manual_sum_rl
    rms3 = torch.sqrt(manual_sum_rl / squared.shape[-1] + 1e-6)
    result3 = test_input / rms3
    
    # 方法4: 分块累加
    chunk_size = 64
    num_chunks = (squared.shape[-1] + chunk_size - 1) // chunk_size
    chunk_sums = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, squared.shape[-1])
        chunk_sum = torch.sum(squared[..., start_idx:end_idx], dim=-1, keepdim=True)
        chunk_sums.append(chunk_sum)
    
    # 累加所有chunk
    total_sum = chunk_sums[0]
    for i in range(1, len(chunk_sums)):
        total_sum = total_sum + chunk_sums[i]
    
    rms4 = torch.sqrt(total_sum / squared.shape[-1] + 1e-6)
    result4 = test_input / rms4
    
    print(f"\n=== RMS计算结果对比 ===")
    print(f"torch.mean方法:     {rms1.mean().item():.15f}")
    print(f"左到右累加方法:     {rms2.mean().item():.15f}")
    print(f"右到左累加方法:     {rms3.mean().item():.15f}")
    print(f"分块累加方法:       {rms4.mean().item():.15f}")
    
    # 计算输出差异
    diff_12 = abs(result1 - result2).max().item()
    diff_13 = abs(result1 - result3).max().item()
    diff_14 = abs(result1 - result4).max().item()
    diff_23 = abs(result2 - result3).max().item()
    
    print(f"\n=== 输出差异分析 ===")
    print(f"torch.mean vs 左到右: {diff_12:.2e}")
    print(f"torch.mean vs 右到左: {diff_13:.2e}")
    print(f"torch.mean vs 分块:   {diff_14:.2e}")
    print(f"左到右 vs 右到左:     {diff_23:.2e}")
    
    if diff_23 > 1e-6:
        print("🎉 成功！不同的累加顺序产生了不同的RMSNorm结果")
    else:
        print("⚠️ 所有方法产生相同结果，可能需要调整数值或方法")

if __name__ == "__main__":
    test_floating_point_non_associativity()
    test_rmsnorm_with_different_orders()
