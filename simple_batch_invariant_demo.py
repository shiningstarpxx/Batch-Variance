#!/usr/bin/env python3
"""
简单的Batch-invariant RMSNorm对比演示

直接展示问题所在和解决方案
"""

import torch
import numpy as np

def demonstrate_batch_variance_problem():
    """演示Batch-variance问题"""
    print("=== Batch-variance问题演示 ===\n")
    
    # 创建相同的输入数据
    torch.manual_seed(42)
    base_input = torch.randn(4, 8)  # 4个元素，8维
    print("📊 基础输入数据:")
    print(base_input)
    print()
    
    # 模拟不同的批处理大小
    batch_sizes = [1, 2, 4]
    
    for batch_size in batch_sizes:
        print(f"🔧 批处理大小: {batch_size}")
        
        # 创建批处理数据（所有样本都相同）
        batch_input = base_input.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # 标准RMSNorm
        std_rms = torch.sqrt(torch.mean(batch_input ** 2, dim=-1, keepdim=True) + 1e-6)
        std_output = batch_input / std_rms
        
        # 模拟Batch-variant RMSNorm（不同的归约顺序）
        if batch_size == 1:
            # 顺序归约
            rms_squared = torch.zeros_like(std_rms)
            for i in range(8):
                rms_squared += batch_input[:, :, i:i+1] ** 2
        elif batch_size == 2:
            # 两两归约
            rms_squared = torch.zeros_like(std_rms)
            for i in range(0, 8, 2):
                if i + 1 < 8:
                    rms_squared += batch_input[:, :, i:i+1] ** 2 + batch_input[:, :, i+1:i+2] ** 2
                else:
                    rms_squared += batch_input[:, :, i:i+1] ** 2
        else:  # batch_size == 4
            # 四四归约
            rms_squared = torch.zeros_like(std_rms)
            for i in range(0, 8, 4):
                chunk_sum = torch.zeros_like(rms_squared)
                for j in range(min(4, 8 - i)):
                    chunk_sum += batch_input[:, :, i+j:i+j+1] ** 2
                rms_squared += chunk_sum
        
        variant_rms = torch.sqrt(rms_squared / 8 + 1e-6)
        variant_output = batch_input / variant_rms
        
        # 计算差异
        diff = torch.max(torch.abs(std_output - variant_output)).item()
        
        print(f"   标准RMSNorm输出 (第一个样本): {std_output[0, 0, :4].numpy()}")
        print(f"   Variant RMSNorm输出 (第一个样本): {variant_output[0, 0, :4].numpy()}")
        print(f"   最大差异: {diff:.2e}")
        print()

def demonstrate_batch_invariant_solution():
    """演示Batch-invariant解决方案"""
    print("=== Batch-invariant解决方案演示 ===\n")
    
    # 创建相同的输入数据
    torch.manual_seed(42)
    base_input = torch.randn(4, 8)
    
    # 模拟不同的批处理大小
    batch_sizes = [1, 2, 4]
    
    print("📊 使用固定归约顺序的RMSNorm:")
    print()
    
    for batch_size in batch_sizes:
        print(f"🔧 批处理大小: {batch_size}")
        
        # 创建批处理数据
        batch_input = base_input.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Batch-invariant RMSNorm（固定归约顺序）
        rms_squared = torch.zeros(batch_size, 4, 1)
        for i in range(8):  # 固定顺序：总是按索引顺序
            rms_squared += batch_input[:, :, i:i+1] ** 2
        
        invariant_rms = torch.sqrt(rms_squared / 8 + 1e-6)
        invariant_output = batch_input / invariant_rms
        
        print(f"   Invariant RMSNorm输出 (第一个样本): {invariant_output[0, 0, :4].numpy()}")
        print()

def compare_solutions():
    """对比解决方案"""
    print("=== 解决方案对比 ===\n")
    
    # 创建相同的输入数据
    torch.manual_seed(42)
    base_input = torch.randn(4, 8)
    
    # 测试批处理大小2
    batch_input = base_input.unsqueeze(0).repeat(2, 1, 1)
    
    # 标准RMSNorm
    std_rms = torch.sqrt(torch.mean(batch_input ** 2, dim=-1, keepdim=True) + 1e-6)
    std_output = batch_input / std_rms
    
    # Batch-variant RMSNorm（两两归约）
    rms_squared_variant = torch.zeros_like(std_rms)
    for i in range(0, 8, 2):
        if i + 1 < 8:
            rms_squared_variant += batch_input[:, :, i:i+1] ** 2 + batch_input[:, :, i+1:i+2] ** 2
        else:
            rms_squared_variant += batch_input[:, :, i:i+1] ** 2
    
    variant_rms = torch.sqrt(rms_squared_variant / 8 + 1e-6)
    variant_output = batch_input / variant_rms
    
    # Batch-invariant RMSNorm（固定顺序）
    rms_squared_invariant = torch.zeros_like(std_rms)
    for i in range(8):  # 固定顺序
        rms_squared_invariant += batch_input[:, :, i:i+1] ** 2
    
    invariant_rms = torch.sqrt(rms_squared_invariant / 8 + 1e-6)
    invariant_output = batch_input / invariant_rms
    
    print("📈 结果对比 (批处理大小=2):")
    print(f"标准RMSNorm:     {std_output[0, 0, :4].numpy()}")
    print(f"Variant RMSNorm: {variant_output[0, 0, :4].numpy()}")
    print(f"Invariant RMSNorm: {invariant_output[0, 0, :4].numpy()}")
    print()
    
    print("🔍 差异分析:")
    std_variant_diff = torch.max(torch.abs(std_output - variant_output)).item()
    std_invariant_diff = torch.max(torch.abs(std_output - invariant_output)).item()
    variant_invariant_diff = torch.max(torch.abs(variant_output - invariant_output)).item()
    
    print(f"标准 vs Variant: {std_variant_diff:.2e}")
    print(f"标准 vs Invariant: {std_invariant_diff:.2e}")
    print(f"Variant vs Invariant: {variant_invariant_diff:.2e}")
    print()
    
    print("💡 关键洞察:")
    print("• Variant方法：不同批处理大小产生不同结果")
    print("• Invariant方法：相同输入总是产生相同结果")
    print("• 差异来源：归约顺序的不同")
    print("• 解决方案：固定归约顺序")

def main():
    """主函数"""
    print("🚀 开始简单的Batch-invariant RMSNorm演示...\n")
    
    # 1. 演示问题
    demonstrate_batch_variance_problem()
    
    # 2. 演示解决方案
    demonstrate_batch_invariant_solution()
    
    # 3. 对比解决方案
    compare_solutions()
    
    print("✅ 演示完成！")

if __name__ == "__main__":
    main()
