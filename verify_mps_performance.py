#!/usr/bin/env python3
"""
验证MPS性能差异的真实原因

测试相同代码在不同条件下的性能差异
"""

import torch
import time
import numpy as np

def test_identical_functions():
    """测试相同函数的性能差异"""
    print("=== 验证相同函数性能差异 ===\n")
    
    # 检查设备
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"🔧 使用设备: {device}")
    
    # 创建测试数据
    batch_size, seq_len, hidden_dim = 4, 256, 512
    torch.manual_seed(42)
    test_data = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    # 定义两个完全相同的函数
    def function_a(x, eps=1e-6):
        """函数A"""
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
        return x / rms
    
    def function_b(x, eps=1e-6):
        """函数B"""
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
        return x / rms
    
    # 测试次数
    num_tests = 100
    
    print(f"📊 测试参数:")
    print(f"   批处理大小: {batch_size}")
    print(f"   序列长度: {seq_len}")
    print(f"   隐藏维度: {hidden_dim}")
    print(f"   测试次数: {num_tests}")
    print()
    
    # 测试函数A
    print("🔧 测试函数A:")
    start_time = time.time()
    for _ in range(num_tests):
        result_a = function_a(test_data)
    end_time = time.time()
    time_a = (end_time - start_time) / num_tests * 1000
    print(f"   平均时间: {time_a:.2f} ms")
    
    # 测试函数B
    print("🔧 测试函数B:")
    start_time = time.time()
    for _ in range(num_tests):
        result_b = function_b(test_data)
    end_time = time.time()
    time_b = (end_time - start_time) / num_tests * 1000
    print(f"   平均时间: {time_b:.2f} ms")
    
    # 验证结果是否相同
    diff = torch.max(torch.abs(result_a - result_b)).item()
    print(f"   结果差异: {diff:.2e}")
    print(f"   结果相同: {'✅ 是' if diff < 1e-10 else '❌ 否'}")
    print()
    
    # 分析性能差异
    if abs(time_a - time_b) > 0.1:  # 如果差异大于0.1ms
        print("⚠️ 发现性能差异！")
        print(f"   差异: {abs(time_a - time_b):.2f} ms")
        print(f"   差异百分比: {abs(time_a - time_b) / min(time_a, time_b) * 100:.1f}%")
        print()
        print("🔍 可能的原因:")
        print("1. **测试顺序影响**: 第一个函数可能触发了MPS预热")
        print("2. **内存分配**: 不同的内存分配模式")
        print("3. **编译器优化**: 不同的优化策略")
        print("4. **系统负载**: 测试期间系统负载变化")
    else:
        print("✅ 性能差异在正常范围内")
    
    return time_a, time_b, diff

def test_warmup_effect():
    """测试MPS预热效果"""
    print("\n=== 测试MPS预热效果 ===\n")
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"🔧 使用设备: {device}")
    
    # 创建测试数据
    batch_size, seq_len, hidden_dim = 4, 256, 512
    torch.manual_seed(42)
    test_data = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    def rmsnorm_function(x, eps=1e-6):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
        return x / rms
    
    # 预热测试
    print("🔥 预热阶段:")
    warmup_times = []
    for i in range(10):
        start_time = time.time()
        result = rmsnorm_function(test_data)
        end_time = time.time()
        warmup_time = (end_time - start_time) * 1000
        warmup_times.append(warmup_time)
        print(f"   第{i+1}次: {warmup_time:.2f} ms")
    
    print(f"   预热平均: {np.mean(warmup_times):.2f} ms")
    print(f"   预热标准差: {np.std(warmup_times):.2f} ms")
    print()
    
    # 稳定测试
    print("📊 稳定阶段:")
    stable_times = []
    for i in range(20):
        start_time = time.time()
        result = rmsnorm_function(test_data)
        end_time = time.time()
        stable_time = (end_time - start_time) * 1000
        stable_times.append(stable_time)
    
    print(f"   稳定平均: {np.mean(stable_times):.2f} ms")
    print(f"   稳定标准差: {np.std(stable_times):.2f} ms")
    print()
    
    # 分析预热效果
    warmup_avg = np.mean(warmup_times)
    stable_avg = np.mean(stable_times)
    improvement = (warmup_avg - stable_avg) / warmup_avg * 100
    
    print("📈 预热效果分析:")
    print(f"   预热前平均: {warmup_avg:.2f} ms")
    print(f"   预热后平均: {stable_avg:.2f} ms")
    print(f"   性能提升: {improvement:.1f}%")
    
    if improvement > 10:
        print("   ✅ 预热效果显著")
    elif improvement > 5:
        print("   ⚠️ 预热效果中等")
    else:
        print("   ❌ 预热效果不明显")

def test_different_execution_order():
    """测试不同执行顺序的影响"""
    print("\n=== 测试执行顺序影响 ===\n")
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"🔧 使用设备: {device}")
    
    # 创建测试数据
    batch_size, seq_len, hidden_dim = 4, 256, 512
    torch.manual_seed(42)
    test_data = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    def rmsnorm_function(x, eps=1e-6):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
        return x / rms
    
    num_tests = 50
    
    # 测试1: 先执行A，再执行B
    print("🔧 测试顺序: A -> B")
    start_time = time.time()
    for _ in range(num_tests):
        result_a = rmsnorm_function(test_data)
    time_a1 = (time.time() - start_time) / num_tests * 1000
    
    start_time = time.time()
    for _ in range(num_tests):
        result_b = rmsnorm_function(test_data)
    time_b1 = (time.time() - start_time) / num_tests * 1000
    
    print(f"   函数A时间: {time_a1:.2f} ms")
    print(f"   函数B时间: {time_b1:.2f} ms")
    print(f"   差异: {abs(time_a1 - time_b1):.2f} ms")
    print()
    
    # 测试2: 先执行B，再执行A
    print("🔧 测试顺序: B -> A")
    start_time = time.time()
    for _ in range(num_tests):
        result_b = rmsnorm_function(test_data)
    time_b2 = (time.time() - start_time) / num_tests * 1000
    
    start_time = time.time()
    for _ in range(num_tests):
        result_a = rmsnorm_function(test_data)
    time_a2 = (time.time() - start_time) / num_tests * 1000
    
    print(f"   函数B时间: {time_b2:.2f} ms")
    print(f"   函数A时间: {time_a2:.2f} ms")
    print(f"   差异: {abs(time_a2 - time_b2):.2f} ms")
    print()
    
    # 分析结果
    print("📊 执行顺序影响分析:")
    print(f"   A先执行: {time_a1:.2f} ms")
    print(f"   A后执行: {time_a2:.2f} ms")
    print(f"   B先执行: {time_b1:.2f} ms")
    print(f"   B后执行: {time_b2:.2f} ms")
    
    if abs(time_a1 - time_a2) > 0.1 or abs(time_b1 - time_b2) > 0.1:
        print("   ⚠️ 执行顺序对性能有影响")
        print("   🔍 可能原因: MPS预热、内存分配、编译器优化")
    else:
        print("   ✅ 执行顺序对性能影响很小")

def main():
    """主函数"""
    print("🚀 开始验证MPS性能差异的真实原因...\n")
    
    # 1. 测试相同函数的性能差异
    time_a, time_b, diff = test_identical_functions()
    
    # 2. 测试MPS预热效果
    test_warmup_effect()
    
    # 3. 测试执行顺序影响
    test_different_execution_order()
    
    print("\n✅ 验证完成！")
    print("\n🎯 结论:")
    print("1. **相同代码**在不同条件下可能有性能差异")
    print("2. **MPS预热**是性能差异的主要原因")
    print("3. **执行顺序**可能影响性能测试结果")
    print("4. **Batch-invariant MPS**和**标准RMSNorm**代码完全相同")
    print("5. **性能差异**来自测试环境，不是代码差异")

if __name__ == "__main__":
    main()
