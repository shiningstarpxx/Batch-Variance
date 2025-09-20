#!/usr/bin/env python3
"""
分析为什么Variant没有差异而Invariant有差异

深入分析并行计算的确定性行为
"""

import torch
import numpy as np
import time
import concurrent.futures
import threading
from typing import List, Dict

class VariantInvariantAnalyzer:
    """Variant vs Invariant差异分析器"""
    
    def __init__(self, device='cpu'):
        """初始化分析器"""
        self.device = torch.device(device)
        print(f"🔧 使用设备: {self.device}")
    
    def create_test_data(self, batch_size: int = 2, seq_len: int = 4, 
                        hidden_dim: int = 8) -> torch.Tensor:
        """创建小规模测试数据便于分析"""
        torch.manual_seed(42)
        test_data = torch.randn(batch_size, seq_len, hidden_dim, device=self.device)
        return test_data
    
    def variant_implementation(self, x: torch.Tensor, chunk_size: int = 2, 
                             num_threads: int = 2, eps: float = 1e-6) -> torch.Tensor:
        """Variant实现：非确定性合并"""
        batch_size, seq_len, hidden_dim = x.shape
        
        # 创建分块索引
        chunk_indices = []
        for i in range(0, hidden_dim, chunk_size):
            end_idx = min(i + chunk_size, hidden_dim)
            chunk_indices.append((i, end_idx))
        
        # 并行计算每个分块
        chunk_results = [None] * len(chunk_indices)
        
        def compute_chunk(chunk_idx, start_idx, end_idx):
            chunk = x[:, :, start_idx:end_idx]
            chunk_sum = torch.sum(chunk ** 2, dim=-1, keepdim=True)
            chunk_results[chunk_idx] = chunk_sum
            print(f"    线程 {threading.current_thread().ident}: 计算分块 {chunk_idx}, 结果: {chunk_sum[0, 0, 0].item():.6f}")
        
        # 使用线程池并行执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i, (start_idx, end_idx) in enumerate(chunk_indices):
                future = executor.submit(compute_chunk, i, start_idx, end_idx)
                futures.append(future)
            
            # 等待所有任务完成
            concurrent.futures.wait(futures)
        
        print(f"    分块结果顺序: {[chunk_results[i][0, 0, 0].item() for i in range(len(chunk_results))]}")
        
        # 合并结果（非确定性顺序）
        rms_squared = torch.zeros(batch_size, seq_len, 1, device=x.device)
        for chunk_sum in chunk_results:
            rms_squared += chunk_sum
        
        rms = torch.sqrt(rms_squared / hidden_dim + eps)
        return x / rms
    
    def invariant_implementation(self, x: torch.Tensor, chunk_size: int = 2, 
                               num_threads: int = 2, eps: float = 1e-6) -> torch.Tensor:
        """Invariant实现：确定性合并"""
        batch_size, seq_len, hidden_dim = x.shape
        
        # 创建分块索引
        chunk_indices = []
        for i in range(0, hidden_dim, chunk_size):
            end_idx = min(i + chunk_size, hidden_dim)
            chunk_indices.append((i, end_idx))
        
        # 并行计算每个分块
        chunk_results = [None] * len(chunk_indices)
        
        def compute_chunk(chunk_idx, start_idx, end_idx):
            chunk = x[:, :, start_idx:end_idx]
            chunk_sum = torch.sum(chunk ** 2, dim=-1, keepdim=True)
            chunk_results[chunk_idx] = chunk_sum
            print(f"    线程 {threading.current_thread().ident}: 计算分块 {chunk_idx}, 结果: {chunk_sum[0, 0, 0].item():.6f}")
        
        # 使用线程池并行执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i, (start_idx, end_idx) in enumerate(chunk_indices):
                future = executor.submit(compute_chunk, i, start_idx, end_idx)
                futures.append(future)
            
            # 等待所有任务完成
            concurrent.futures.wait(futures)
        
        print(f"    分块结果顺序: {[chunk_results[i][0, 0, 0].item() for i in range(len(chunk_results))]}")
        
        # 合并结果（确定性顺序：按索引顺序）
        rms_squared = torch.zeros(batch_size, seq_len, 1, device=x.device)
        for i in range(len(chunk_results)):
            if chunk_results[i] is not None:
                rms_squared += chunk_results[i]
                print(f"    按索引 {i} 合并: {chunk_results[i][0, 0, 0].item():.6f}")
        
        rms = torch.sqrt(rms_squared / hidden_dim + eps)
        return x / rms
    
    def analyze_determinism(self) -> None:
        """分析确定性行为"""
        print("=== 分析Variant vs Invariant的确定性行为 ===\n")
        
        # 创建测试数据
        test_data = self.create_test_data()
        print(f"📊 测试数据形状: {test_data.shape}")
        print(f"   数据: {test_data[0, 0, :].numpy()}")
        print()
        
        # 测试次数
        num_tests = 5
        
        print("🔧 测试Variant实现（非确定性合并）:")
        variant_outputs = []
        for i in range(num_tests):
            print(f"  第{i+1}次执行:")
            output = self.variant_implementation(test_data)
            variant_outputs.append(output)
            print(f"    最终结果: {output[0, 0, 0].item():.8f}")
            print()
        
        print("🔧 测试Invariant实现（确定性合并）:")
        invariant_outputs = []
        for i in range(num_tests):
            print(f"  第{i+1}次执行:")
            output = self.invariant_implementation(test_data)
            invariant_outputs.append(output)
            print(f"    最终结果: {output[0, 0, 0].item():.8f}")
            print()
        
        # 分析差异
        print("📈 差异分析:")
        
        # Variant差异
        variant_ref = variant_outputs[0]
        variant_diffs = []
        for output in variant_outputs[1:]:
            diff = torch.max(torch.abs(output - variant_ref)).item()
            variant_diffs.append(diff)
        
        print(f"  Variant最大差异: {max(variant_diffs):.2e}")
        print(f"  Variant平均差异: {np.mean(variant_diffs):.2e}")
        
        # Invariant差异
        invariant_ref = invariant_outputs[0]
        invariant_diffs = []
        for output in invariant_outputs[1:]:
            diff = torch.max(torch.abs(output - invariant_ref)).item()
            invariant_diffs.append(diff)
        
        print(f"  Invariant最大差异: {max(invariant_diffs):.2e}")
        print(f"  Invariant平均差异: {np.mean(invariant_diffs):.2e}")
        print()
        
        # 分析原因
        print("🔍 原因分析:")
        if max(variant_diffs) < 1e-10:
            print("  ✅ Variant实现实际上是确定性的")
            print("     可能原因:")
            print("     1. 线程池执行顺序相对稳定")
            print("     2. 小规模数据，并行开销大于收益")
            print("     3. 系统调度相对稳定")
        else:
            print("  ❌ Variant实现确实是非确定性的")
        
        if max(invariant_diffs) < 1e-10:
            print("  ✅ Invariant实现是确定性的")
        else:
            print("  ❌ Invariant实现出现非确定性")
            print("     可能原因:")
            print("     1. 线程竞争导致计算顺序变化")
            print("     2. 浮点运算的微小差异")
            print("     3. 内存访问模式的影响")
    
    def test_larger_scale(self) -> None:
        """测试更大规模的数据"""
        print("\n=== 测试更大规模数据 ===\n")
        
        # 创建更大规模的测试数据
        test_data = self.create_test_data(batch_size=4, seq_len=256, hidden_dim=512)
        print(f"📊 测试数据形状: {test_data.shape}")
        
        # 测试次数
        num_tests = 10
        
        print("🔧 测试Variant实现:")
        variant_outputs = []
        for i in range(num_tests):
            output = self.variant_implementation(test_data, chunk_size=64, num_threads=4)
            variant_outputs.append(output)
            if i < 3:  # 只显示前3次的结果
                print(f"  第{i+1}次结果: {output[0, 0, 0].item():.8f}")
        
        print("🔧 测试Invariant实现:")
        invariant_outputs = []
        for i in range(num_tests):
            output = self.invariant_implementation(test_data, chunk_size=64, num_threads=4)
            invariant_outputs.append(output)
            if i < 3:  # 只显示前3次的结果
                print(f"  第{i+1}次结果: {output[0, 0, 0].item():.8f}")
        
        # 分析差异
        print("\n📈 大规模数据差异分析:")
        
        # Variant差异
        variant_ref = variant_outputs[0]
        variant_diffs = []
        for output in variant_outputs[1:]:
            diff = torch.max(torch.abs(output - variant_ref)).item()
            variant_diffs.append(diff)
        
        print(f"  Variant最大差异: {max(variant_diffs):.2e}")
        print(f"  Variant平均差异: {np.mean(variant_diffs):.2e}")
        
        # Invariant差异
        invariant_ref = invariant_outputs[0]
        invariant_diffs = []
        for output in invariant_outputs[1:]:
            diff = torch.max(torch.abs(output - invariant_ref)).item()
            invariant_diffs.append(diff)
        
        print(f"  Invariant最大差异: {max(invariant_diffs):.2e}")
        print(f"  Invariant平均差异: {np.mean(invariant_diffs):.2e}")
        
        # 结论
        print("\n🎯 结论:")
        if max(variant_diffs) > max(invariant_diffs):
            print("  ✅ 大规模数据下，Variant确实比Invariant更不稳定")
        elif max(variant_diffs) < max(invariant_diffs):
            print("  ⚠️ 大规模数据下，Invariant比Variant更不稳定")
        else:
            print("  ✅ 大规模数据下，两者稳定性相当")

def main():
    """主函数"""
    analyzer = VariantInvariantAnalyzer(device='cpu')
    
    # 1. 分析小规模数据的确定性行为
    analyzer.analyze_determinism()
    
    # 2. 测试大规模数据
    analyzer.test_larger_scale()
    
    print("\n✅ 分析完成！")

if __name__ == "__main__":
    main()
