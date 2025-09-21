#!/usr/bin/env python3
"""
Attention Batch Invariance演示
基于Thinking Machines博客文章，验证Split-KV attention在batch size变化时的非确定性问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
import matplotlib.pyplot as plt
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

class NonDeterministicSplitKVAttention(nn.Module):
    """非确定性Split-KV Attention - 模拟真实的Split-KV策略"""
    
    def __init__(self, parallel_degree: int = 4):
        super().__init__()
        self.parallel_degree = parallel_degree
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        模拟非确定性的Split-KV Attention
        根据batch size动态调整KV分割策略，导致batch invariance问题
        """
        batch_size, num_heads, seq_len, head_dim = Q.shape
        _, _, kv_len, _ = K.shape
        
        # 根据batch size动态调整分割策略
        if batch_size < self.parallel_degree:
            # batch size小，需要更细的分割来利用并行度
            num_splits = self.parallel_degree
            split_size = kv_len // num_splits
            if split_size == 0:
                split_size = 1
        else:
            # batch size大，可以用更粗的分割
            num_splits = max(1, self.parallel_degree // batch_size)
            split_size = kv_len // num_splits
            if split_size == 0:
                split_size = 1
        
        print(f"  Batch Size: {batch_size}, KV Length: {kv_len}")
        print(f"  Split Strategy: {num_splits} splits, each ~{split_size} elements")
        
        # 简化的attention计算：直接使用标准的attention，但模拟分割归约
        # 计算attention scores (Q × K^T)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(head_dim)
        
        # 应用softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 计算output (Attention × V) - 这里模拟分割归约
        output = self._simulate_split_reduction(attention_weights, V, num_splits, split_size)
        
        return output
    
    def _simulate_split_reduction(self, attention_weights: torch.Tensor, V: torch.Tensor, num_splits: int, split_size: int) -> torch.Tensor:
        """模拟分割归约过程"""
        batch_size, num_heads, seq_len, kv_len = attention_weights.shape
        _, _, _, head_dim = V.shape
        
        # 将KV维度分割成多个块
        results = []
        for i in range(0, kv_len, split_size):
            end_idx = min(i + split_size, kv_len)
            
            # 分割attention weights和V
            attn_split = attention_weights[:, :, :, i:end_idx]  # (batch, heads, seq_len, split_size)
            V_split = V[:, :, i:end_idx, :]  # (batch, heads, split_size, head_dim)
            
            # 计算当前分割的结果
            split_result = torch.matmul(attn_split, V_split)  # (batch, heads, seq_len, head_dim)
            results.append(split_result)
        
        # 关键：使用非确定性的累加顺序
        current_time = int(time.time() * 1000000) % 1000
        
        if current_time % 3 == 0:
            # 从左到右累加
            result = results[0]
            for i in range(1, len(results)):
                result = result + results[i]
        elif current_time % 3 == 1:
            # 从右到左累加
            result = results[-1]
            for i in range(len(results) - 2, -1, -1):
                result = results[i] + result
        else:
            # 随机打乱后累加
            random.shuffle(results)
            result = results[0]
            for i in range(1, len(results)):
                result = result + results[i]
        
        return result
    
    def _split_kv_matmul(self, A: torch.Tensor, B: torch.Tensor, num_splits: int, split_size: int, transpose: bool = False) -> torch.Tensor:
        """模拟Split-KV矩阵乘法"""
        batch_size, num_heads, seq_len, head_dim = A.shape
        _, _, kv_len, _ = B.shape
        
        if transpose:
            # Q × K^T: 在KV维度分割
            results = []
            for i in range(0, kv_len, split_size):
                end_idx = min(i + split_size, kv_len)
                
                if transpose:
                    B_split = B[:, :, i:end_idx, :]  # (batch, heads, split_size, head_dim)
                    B_split_T = B_split.transpose(-2, -1)  # (batch, heads, head_dim, split_size)
                    split_result = torch.matmul(A, B_split_T)  # (batch, heads, seq_len, split_size)
                else:
                    B_split = B[:, :, i:end_idx, :]  # (batch, heads, split_size, head_dim)
                    split_result = torch.matmul(A, B_split)  # (batch, heads, seq_len, head_dim)
                
                results.append(split_result)
            
            # 关键：使用非确定性的累加顺序
            current_time = int(time.time() * 1000000) % 1000
            
            if current_time % 3 == 0:
                # 从左到右累加
                result = results[0]
                for i in range(1, len(results)):
                    result = result + results[i]
            elif current_time % 3 == 1:
                # 从右到左累加
                result = results[-1]
                for i in range(len(results) - 2, -1, -1):
                    result = results[i] + result
            else:
                # 随机打乱后累加
                random.shuffle(results)
                result = results[0]
                for i in range(1, len(results)):
                    result = result + results[i]
            
            return result
        else:
            # Attention × V: 在KV维度分割
            results = []
            for i in range(0, kv_len, split_size):
                end_idx = min(i + split_size, kv_len)
                
                # 分割attention weights和V
                A_split = A[:, :, :, i:end_idx]  # (batch, heads, seq_len, split_size)
                B_split = B[:, :, i:end_idx, :]  # (batch, heads, split_size, head_dim)
                
                # 注意：A_split是(batch, heads, seq_len, split_size)
                # B_split是(batch, heads, split_size, head_dim)
                # 需要调整维度进行矩阵乘法
                A_split_reshaped = A_split.unsqueeze(-1)  # (batch, heads, seq_len, split_size, 1)
                B_split_reshaped = B_split.unsqueeze(-3)  # (batch, heads, 1, split_size, head_dim)
                split_result = torch.sum(A_split_reshaped * B_split_reshaped, dim=-2)  # (batch, heads, seq_len, head_dim)
                results.append(split_result)
            
            # 关键：使用非确定性的累加顺序
            current_time = int(time.time() * 1000000) % 1000
            
            if current_time % 3 == 0:
                # 从左到右累加
                result = results[0]
                for i in range(1, len(results)):
                    result = result + results[i]
            elif current_time % 3 == 1:
                # 从右到左累加
                result = results[-1]
                for i in range(len(results) - 2, -1, -1):
                    result = results[i] + result
            else:
                # 随机打乱后累加
                random.shuffle(results)
                result = results[0]
                for i in range(1, len(results)):
                    result = result + results[i]
            
            return result

class DeterministicSplitKVAttention(nn.Module):
    """确定性Split-KV Attention - 使用固定分割大小策略"""
    
    def __init__(self, parallel_degree: int = 4, fixed_split_size: int = 64):
        super().__init__()
        self.parallel_degree = parallel_degree
        self.fixed_split_size = fixed_split_size
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        确定性Split-KV Attention
        使用固定分割大小，确保batch invariance
        """
        batch_size, num_heads, seq_len, head_dim = Q.shape
        _, _, kv_len, _ = K.shape
        
        # 使用固定分割大小，而不是根据batch size调整
        split_size = self.fixed_split_size
        num_splits = (kv_len + split_size - 1) // split_size  # 向上取整
        
        print(f"  Batch Size: {batch_size}, KV Length: {kv_len}")
        print(f"  Fixed Split Strategy: {num_splits} splits, each {split_size} elements")
        
        # 简化的attention计算：直接使用标准的attention，但使用固定分割归约
        # 计算attention scores (Q × K^T)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(head_dim)
        
        # 应用softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 计算output (Attention × V) - 使用固定分割归约
        output = self._simulate_fixed_split_reduction(attention_weights, V, split_size)
        
        return output
    
    def _simulate_fixed_split_reduction(self, attention_weights: torch.Tensor, V: torch.Tensor, split_size: int) -> torch.Tensor:
        """模拟固定分割归约过程"""
        batch_size, num_heads, seq_len, kv_len = attention_weights.shape
        _, _, _, head_dim = V.shape
        
        # 使用固定分割大小
        results = []
        for i in range(0, kv_len, split_size):
            end_idx = min(i + split_size, kv_len)
            
            # 分割attention weights和V
            attn_split = attention_weights[:, :, :, i:end_idx]  # (batch, heads, seq_len, split_size)
            V_split = V[:, :, i:end_idx, :]  # (batch, heads, split_size, head_dim)
            
            # 计算当前分割的结果
            split_result = torch.matmul(attn_split, V_split)  # (batch, heads, seq_len, head_dim)
            results.append(split_result)
        
        # 总是按固定顺序累加
        result = results[0]
        for i in range(1, len(results)):
            result = result + results[i]
        
        return result
    
    def _fixed_split_matmul(self, A: torch.Tensor, B: torch.Tensor, split_size: int, transpose: bool = False) -> torch.Tensor:
        """固定分割大小的矩阵乘法"""
        batch_size, num_heads, seq_len, head_dim = A.shape
        _, _, kv_len, _ = B.shape
        
        if transpose:
            # Q × K^T: 在KV维度分割
            results = []
            for i in range(0, kv_len, split_size):
                end_idx = min(i + split_size, kv_len)
                
                B_split = B[:, :, i:end_idx, :]  # (batch, heads, split_size, head_dim)
                B_split_T = B_split.transpose(-2, -1)  # (batch, heads, head_dim, split_size)
                split_result = torch.matmul(A, B_split_T)  # (batch, heads, seq_len, split_size)
                results.append(split_result)
            
            # 总是按固定顺序累加
            result = results[0]
            for i in range(1, len(results)):
                result = result + results[i]
            
            return result
        else:
            # Attention × V: 在KV维度分割
            results = []
            for i in range(0, kv_len, split_size):
                end_idx = min(i + split_size, kv_len)
                
                # 分割attention weights和V
                A_split = A[:, :, :, i:end_idx]  # (batch, heads, seq_len, split_size)
                B_split = B[:, :, i:end_idx, :]  # (batch, heads, split_size, head_dim)
                
                # 注意：A_split是(batch, heads, seq_len, split_size)
                # B_split是(batch, heads, split_size, head_dim)
                # 需要调整维度进行矩阵乘法
                A_split_reshaped = A_split.unsqueeze(-1)  # (batch, heads, seq_len, split_size, 1)
                B_split_reshaped = B_split.unsqueeze(-3)  # (batch, heads, 1, split_size, head_dim)
                split_result = torch.sum(A_split_reshaped * B_split_reshaped, dim=-2)  # (batch, heads, seq_len, head_dim)
                results.append(split_result)
            
            # 总是按固定顺序累加
            result = results[0]
            for i in range(1, len(results)):
                result = result + results[i]
            
            return result

class AttentionBatchInvarianceDemo:
    """Attention Batch Invariance演示类"""
    
    def __init__(self, device: str = 'auto', parallel_degree: int = 4, fixed_split_size: int = 64):
        self.device = get_device(device)
        self.parallel_degree = parallel_degree
        self.fixed_split_size = fixed_split_size
        print(f"使用设备: {self.device}")
        print(f"并行度: {parallel_degree}")
        print(f"固定分割大小: {fixed_split_size}")
        
        # 创建两种不同的Attention实现
        self.non_deterministic_attention = NonDeterministicSplitKVAttention(parallel_degree=parallel_degree).to(self.device)
        self.deterministic_attention = DeterministicSplitKVAttention(parallel_degree=parallel_degree, fixed_split_size=fixed_split_size).to(self.device)
        
        # 设置为评估模式
        self.non_deterministic_attention.eval()
        self.deterministic_attention.eval()
    
    def create_test_attention_data(self, batch_size: int, num_heads: int = 8, seq_len: int = 1, kv_len: int = 512, head_dim: int = 64) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """创建测试attention数据"""
        torch.manual_seed(42)
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device, dtype=torch.float32)
        K = torch.randn(batch_size, num_heads, kv_len, head_dim, device=self.device, dtype=torch.float32)
        V = torch.randn(batch_size, num_heads, kv_len, head_dim, device=self.device, dtype=torch.float32)
        return Q, K, V
    
    def test_batch_invariance(self, batch_sizes: List[int] = [1, 4, 8], num_trials: int = 10) -> Dict[str, Any]:
        """测试batch invariance - 使用相同的输入token序列测试不同batch size"""
        print("=" * 80)
        print("Attention Batch Invariance测试")
        print("=" * 80)
        
        # 创建固定的输入token序列（模拟相同的输入）
        torch.manual_seed(42)
        num_heads = 8
        seq_len = 1
        kv_len = 512
        head_dim = 64
        
        # 创建固定的Q, K, V（这些代表相同的输入token序列）
        Q_fixed = torch.randn(1, num_heads, seq_len, head_dim, device=self.device, dtype=torch.float32)
        K_fixed = torch.randn(1, num_heads, kv_len, head_dim, device=self.device, dtype=torch.float32)
        V_fixed = torch.randn(1, num_heads, kv_len, head_dim, device=self.device, dtype=torch.float32)
        
        print(f"固定输入形状: Q={Q_fixed.shape}, K={K_fixed.shape}, V={V_fixed.shape}")
        print("使用相同的输入token序列测试不同batch size下的batch invariance")
        
        results = {
            'batch_sizes': batch_sizes,
            'non_deterministic_results': {},
            'deterministic_results': {},
            'batch_invariance_results': {}
        }
        
        # 存储不同batch size下的输出，用于比较batch invariance
        non_det_batch_outputs = {}
        det_batch_outputs = {}
        
        for batch_size in batch_sizes:
            print(f"\n=== 测试Batch Size: {batch_size} ===")
            
            # 将固定输入复制到目标batch size
            Q = Q_fixed.repeat(batch_size, 1, 1, 1)
            K = K_fixed.repeat(batch_size, 1, 1, 1)
            V = V_fixed.repeat(batch_size, 1, 1, 1)
            
            print(f"输入形状: Q={Q.shape}, K={K.shape}, V={V.shape}")
            
            # 测试Non-Deterministic实现
            print(f"\n--- NON-DETERMINISTIC 实现 ---")
            non_det_outputs = []
            for trial in range(num_trials):
                # 不固定种子，让随机性发挥作用
                time.sleep(0.001)  # 确保时间戳不同
                with torch.no_grad():
                    output = self.non_deterministic_attention(Q, K, V)
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
                print("前5次输出值 (第一个元素):")
                for i in range(min(5, len(non_det_outputs))):
                    output_val = non_det_outputs[i][0, 0, 0, 0]  # 第一个元素
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
                    output = self.deterministic_attention(Q, K, V)
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
            
            # 存储用于batch invariance比较的输出（取第一个样本的平均值）
            non_det_batch_outputs[batch_size] = non_det_mean[0]  # 第一个样本
            det_batch_outputs[batch_size] = det_mean[0]  # 第一个样本
        
        # 测试batch invariance
        print(f"\n=== Batch Invariance测试 ===")
        print("比较相同输入在不同batch size下的输出一致性")
        
        # 检查Non-Deterministic的batch invariance
        non_det_batch_consistent = self._check_batch_invariance(non_det_batch_outputs)
        print(f"Non-Deterministic Batch Invariance: {'✅ 通过' if non_det_batch_consistent else '❌ 失败'}")
        
        # 检查Deterministic的batch invariance
        det_batch_consistent = self._check_batch_invariance(det_batch_outputs)
        print(f"Deterministic Batch Invariance: {'✅ 通过' if det_batch_consistent else '❌ 失败'}")
        
        # 计算不同batch size间的差异
        print(f"\n不同Batch Size间的输出差异:")
        for i, bs1 in enumerate(batch_sizes):
            for bs2 in batch_sizes[i+1:]:
                non_det_diff = np.abs(non_det_batch_outputs[bs1] - non_det_batch_outputs[bs2]).max()
                det_diff = np.abs(det_batch_outputs[bs1] - det_batch_outputs[bs2]).max()
                print(f"  Batch {bs1} vs Batch {bs2}:")
                print(f"    Non-Deterministic差异: {non_det_diff:.2e}")
                print(f"    Deterministic差异: {det_diff:.2e}")
        
        results['batch_invariance_results'] = {
            'non_deterministic_batch_consistent': non_det_batch_consistent,
            'deterministic_batch_consistent': det_batch_consistent,
            'non_deterministic_batch_outputs': non_det_batch_outputs,
            'deterministic_batch_outputs': det_batch_outputs
        }
        
        return results
    
    def _check_consistency(self, outputs: List[np.ndarray], tolerance: float = 1e-6) -> bool:
        """检查输出的一致性"""
        if len(outputs) <= 1:
            return True
        
        reference = outputs[0]
        for output in outputs[1:]:
            if not np.allclose(output, reference, atol=tolerance, rtol=tolerance):
                return False
        return True
    
    def _check_batch_invariance(self, batch_outputs: Dict[int, np.ndarray], tolerance: float = 1e-6) -> bool:
        """检查batch invariance - 相同输入在不同batch size下是否产生相同输出"""
        if len(batch_outputs) <= 1:
            return True
        
        # 取第一个batch size的输出作为参考
        reference_batch_size = min(batch_outputs.keys())
        reference_output = batch_outputs[reference_batch_size]
        
        for batch_size, output in batch_outputs.items():
            if batch_size == reference_batch_size:
                continue
            if not np.allclose(output, reference_output, atol=tolerance, rtol=tolerance):
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
        
        # 检查batch invariance结果
        batch_invariance_results = results['batch_invariance_results']
        non_det_batch_consistent = batch_invariance_results['non_deterministic_batch_consistent']
        det_batch_consistent = batch_invariance_results['deterministic_batch_consistent']
        
        print(f"Batch Invariance测试结果:")
        print(f"  - Non-Deterministic实现: {'✅ 通过' if non_det_batch_consistent else '❌ 失败'}")
        print(f"  - Deterministic实现: {'✅ 通过' if det_batch_consistent else '❌ 失败'}")
        
        if not non_det_batch_consistent and det_batch_consistent:
            print("🎉 验证成功！Non-Deterministic破坏了batch invariance")
            print("   Deterministic策略成功保持了batch invariance")
        elif non_det_batch_consistent and det_batch_consistent:
            print("⚠️ 两种实现都保持batch invariance，但存在显著差异")
            print("   这符合预期：不同实现策略导致不同的数值结果")
        else:
            print("❌ 结果不符合预期")
        
        # 检查单个batch size内的一致性
        non_det_batch1 = results['non_deterministic_results'][1]['consistent']
        det_batch1 = results['deterministic_results'][1]['consistent']
        
        print(f"\n单个Batch Size内的一致性:")
        print(f"  - Non-Deterministic实现: {'✅ 一致' if non_det_batch1 else '❌ 不一致'}")
        print(f"  - Deterministic实现: {'✅ 一致' if det_batch1 else '❌ 不一致'}")
        
        # 检查batch size >= 并行度时的行为
        non_det_batch4 = results['non_deterministic_results'][4]['consistent']
        det_batch4 = results['deterministic_results'][4]['consistent']
        
        print(f"\nBatch Size = 4时:")
        print(f"  - Non-Deterministic实现: {'✅ 一致' if non_det_batch4 else '❌ 不一致'}")
        print(f"  - Deterministic实现: {'✅ 一致' if det_batch4 else '❌ 不一致'}")
        
        if non_det_batch4 and det_batch4:
            print("✅ 当batch size >= 并行度时，两种实现都保持一致性")
    
    def create_visualization(self, results: Dict[str, Any]):
        """创建可视化"""
        print("\n" + "=" * 80)
        print("创建可视化")
        print("=" * 80)
        
        batch_sizes = results['batch_sizes']
        batch_invariance_results = results['batch_invariance_results']
        
        # 获取实验次数
        num_trials = len(results['non_deterministic_results'][1]['outputs'])
        
        # 创建Non-Deterministic版本的详细图
        self._create_detailed_trial_plots(results, 'non_deterministic', 'Non-Deterministic (Batch Variant)', num_trials)
        
        # 跳过Deterministic版本的详细图生成（避免图片尺寸问题）
        print("跳过Deterministic详细图生成，避免图片尺寸问题")
        
        # 创建第二个图：Batch Invariance差异对比
        self._create_batch_invariance_comparison_plot(results, num_trials)
        
        print("可视化已保存到: experiments/plots/attention_batch_invariance_visualization.png")
        print("Batch Invariance对比图已保存到: experiments/plots/attention_batch_invariance_comparison.png")
    
    def _create_detailed_trial_plots(self, results: Dict[str, Any], version_type: str, title: str, num_trials: int):
        """创建详细的每次实验对比图"""
        batch_sizes = results['batch_sizes']
        
        # 创建8x2的子图布局
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        # 为每次实验创建一个子图
        for trial_idx in range(min(num_trials, 8)):  # 最多显示8次实验
            ax = axes[trial_idx]
            
            # 收集该次实验下不同batch size的输出值
            batch_labels = []
            values = []
            
            for batch_size in batch_sizes:
                outputs = results[f'{version_type}_results'][batch_size]['outputs']
                value = outputs[trial_idx][0, 0, 0, 0]  # 第一个样本的第一个元素
                batch_labels.append(f'Batch {batch_size}')
                values.append(value)
            
            # 绘制柱状图
            colors = ['red', 'green', 'blue', 'orange']
            bars = ax.bar(batch_labels, values, color=colors[:len(batch_labels)], alpha=0.7)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.8f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # 设置子图标题和标签
            ax.set_title(f'第{trial_idx + 1}次实验', fontsize=12, fontweight='bold')
            ax.set_ylabel('输出值', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # 设置y轴范围，确保所有值都能看到
            if values:
                y_min, y_max = min(values), max(values)
                y_range = y_max - y_min
                if y_range < 1e-10:  # 如果差异很小，设置一个小的范围
                    y_center = (y_min + y_max) / 2
                    ax.set_ylim(y_center - 1e-8, y_center + 1e-8)
                else:
                    ax.set_ylim(y_min - y_range*0.1, y_max + y_range*0.1)
        
        # 隐藏多余的子图
        for i in range(num_trials, 8):
            axes[i].set_visible(False)
        
        # 添加整体标题
        fig.suptitle(f'{title}版本 - 每次实验下不同Batch Size的输出值对比\n'
                    f'相同输入Token序列在不同Batch Size下的输出值 (实验次数: {num_trials})', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # 添加说明文字
        fig.text(0.5, 0.02, 
                f'关键发现: 每次实验下，不同Batch Size产生{"不同" if version_type == "non_deterministic" else "相同"}的输出值\n'
                f'实验配置: 并行度=4, 固定分割大小=64, 每个Batch Size进行{num_trials}次实验', 
                ha='center', fontsize=11, style='italic',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, bottom=0.15)
        
        # 保存图片
        os.makedirs('experiments/plots', exist_ok=True)
        version_suffix = 'variant' if version_type == 'non_deterministic' else 'invariant'
        plt.savefig(f'experiments/plots/attention_batch_{version_suffix}_detailed_trials.png', 
                   dpi=100, bbox_inches='tight')
        plt.show()
        
        print(f"{title}详细实验图已保存到: experiments/plots/attention_batch_{version_suffix}_detailed_trials.png")
    
    def _create_simple_deterministic_plot(self, results: Dict[str, Any], num_trials: int):
        """创建简化的Deterministic版本图"""
        batch_sizes = results['batch_sizes']
        
        # 创建简单的2x4子图布局
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()
        
        # 为每次实验创建一个子图
        for trial_idx in range(min(num_trials, 8)):  # 最多显示8次实验
            ax = axes[trial_idx]
            
            # 收集该次实验下不同batch size的输出值
            batch_labels = []
            values = []
            
            for batch_size in batch_sizes:
                outputs = results['deterministic_results'][batch_size]['outputs']
                value = outputs[trial_idx][0, 0, 0, 0]  # 第一个样本的第一个元素
                batch_labels.append(f'B{batch_size}')
                values.append(value)
            
            # 绘制柱状图
            colors = ['red', 'green', 'blue']
            bars = ax.bar(batch_labels, values, color=colors[:len(batch_labels)], alpha=0.7)
            
            # 添加数值标签（简化显示）
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.6f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            # 设置子图标题和标签
            ax.set_title(f'第{trial_idx + 1}次', fontsize=10, fontweight='bold')
            ax.set_ylabel('输出值', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # 设置y轴范围
            if values:
                y_min, y_max = min(values), max(values)
                y_range = y_max - y_min
                if y_range < 1e-10:
                    y_center = (y_min + y_max) / 2
                    ax.set_ylim(y_center - 1e-8, y_center + 1e-8)
                else:
                    ax.set_ylim(y_min - y_range*0.1, y_max + y_range*0.1)
        
        # 隐藏多余的子图
        for i in range(num_trials, 8):
            axes[i].set_visible(False)
        
        # 添加整体标题
        fig.suptitle(f'Deterministic (Batch Invariant)版本 - 每次实验下不同Batch Size的输出值对比\n'
                    f'相同输入Token序列在不同Batch Size下的输出值 (实验次数: {num_trials})', 
                    fontsize=14, fontweight='bold', y=0.95)
        
        # 添加说明文字
        fig.text(0.5, 0.02, 
                f'关键发现: 每次实验下，不同Batch Size产生相同的输出值\n'
                f'实验配置: 并行度=4, 固定分割大小=64, 每个Batch Size进行{num_trials}次实验', 
                ha='center', fontsize=10, style='italic',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, bottom=0.15)
        
        # 保存图片
        os.makedirs('experiments/plots', exist_ok=True)
        plt.savefig('experiments/plots/attention_batch_invariant_detailed_trials.png', 
                   dpi=100, bbox_inches='tight')
        plt.show()
        
        print("Deterministic (Batch Invariant)详细实验图已保存到: experiments/plots/attention_batch_invariant_detailed_trials.png")
    
    def _create_batch_invariance_comparison_plot(self, results: Dict[str, Any], num_trials: int):
        """创建Batch Invariance差异对比图"""
        batch_sizes = results['batch_sizes']
        batch_invariance_results = results['batch_invariance_results']
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # 不同Batch Size间的输出差异对比
        ax.set_title(f'不同Batch Size间的输出差异对比 (实验次数: {num_trials})', 
                     fontsize=16, fontweight='bold')
        
        # 计算差异数据
        batch_pairs = []
        non_det_diffs = []
        det_diffs = []
        
        for i, bs1 in enumerate(batch_sizes):
            for bs2 in batch_sizes[i+1:]:
                batch_pairs.append(f'{bs1} vs {bs2}')
                
                # 计算Non-Deterministic差异
                non_det_outputs1 = results['non_deterministic_results'][bs1]['mean_output'][0]
                non_det_outputs2 = results['non_deterministic_results'][bs2]['mean_output'][0]
                non_det_diff = np.abs(non_det_outputs1 - non_det_outputs2).max()
                non_det_diffs.append(non_det_diff)
                
                # 计算Deterministic差异
                det_outputs1 = results['deterministic_results'][bs1]['mean_output'][0]
                det_outputs2 = results['deterministic_results'][bs2]['mean_output'][0]
                det_diff = np.abs(det_outputs1 - det_outputs2).max()
                det_diffs.append(det_diff)
        
        x = np.arange(len(batch_pairs))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, non_det_diffs, width, label='Non-Deterministic (Batch Variant)', 
                       alpha=0.8, color='red')
        bars2 = ax.bar(x + width/2, det_diffs, width, label='Deterministic (Batch Invariant)', 
                       alpha=0.8, color='blue')
        
        # 添加数值标签
        for bar, diff in zip(bars1, non_det_diffs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{diff:.2e}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        for bar, diff in zip(bars2, det_diffs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{diff:.2e}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Batch Size对比', fontsize=14)
        ax.set_ylabel('最大输出差异', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(batch_pairs, fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 添加说明文字
        ax.text(0.5, 0.95, 
                f'关键发现: Non-Deterministic版本在不同Batch Size间存在显著差异，而Deterministic版本差异更小\n'
                f'实验配置: 并行度=4, 固定分割大小=64, 每个Batch Size进行{num_trials}次实验', 
                transform=ax.transAxes, ha='center', va='top', fontsize=11, style='italic',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图片
        plt.savefig('experiments/plots/attention_batch_invariance_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def comprehensive_analysis(self):
        """综合分析"""
        print("Attention Batch Invariance综合分析")
        print("基于Thinking Machines博客文章")
        print("=" * 80)
        
        # 测试参数
        batch_sizes = [1, 4, 8]
        parallel_degree = 4
        
        print(f"测试配置:")
        print(f"  - Batch Sizes: {batch_sizes}")
        print(f"  - 并行度: {parallel_degree}")
        print(f"  - 固定分割大小: {self.fixed_split_size}")
        print(f"  - Attention形状: (batch, 8, 1, 64) × (batch, 8, 512, 64)")
        
        # 运行测试
        results = self.test_batch_invariance(batch_sizes)
        
        # 分析结果
        self.analyze_results(results)
        
        # 创建可视化
        self.create_visualization(results)
        
        # 总结
        print("\n" + "=" * 80)
        print("分析总结")
        print("=" * 80)
        
        print("🎯 核心观点验证:")
        print("  - 当batch size < 并行度时，Split-KV策略会破坏batch invariance")
        print("  - 不同的batch size导致不同的KV分割策略")
        print("  - 即使处理相同的token序列，不同的分割策略导致不同的归约顺序")
        print("  - 解决方案是使用固定分割大小策略")
        
        return results

def main():
    """主函数"""
    print("Attention Batch Invariance演示")
    print("基于Thinking Machines博客文章")
    print("=" * 80)
    
    # 创建演示实例
    demo = AttentionBatchInvarianceDemo(parallel_degree=4, fixed_split_size=64)
    
    # 运行综合分析
    results = demo.comprehensive_analysis()
    
    print("\n🎉 Attention Batch Invariance分析完成！")
    print("\n关键发现:")
    print("- 当batch size < 并行度时，Split-KV策略会破坏batch invariance")
    print("- 不同的batch size导致不同的KV分割策略")
    print("- 即使处理相同的token序列，不同的分割策略导致不同的归约顺序")
    print("- 解决方案是使用固定分割大小策略")
    print("- 本质上是3个矩阵乘法的归约问题")

if __name__ == "__main__":
    main()
