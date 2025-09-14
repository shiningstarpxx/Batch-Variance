"""
批处理不变性解决方案模块

这个模块实现了批处理不变性的解决方案，确保不同批处理大小下的数值结果保持一致。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional
import time
import random

# 导入字体配置
try:
    from .font_config import setup_chinese_fonts
except ImportError:
    from font_config import setup_chinese_fonts

# 设置中文字体
setup_chinese_fonts()

class BatchInvariantOps:
    """批处理不变性操作类"""
    
    def __init__(self, device='cpu'):
        self.device = device
    
    def batch_invariant_sum(self, x: torch.Tensor, dim: int = -1, 
                           fixed_chunk_size: int = 64) -> torch.Tensor:
        """批处理不变性求和"""
        if x.size(dim) <= fixed_chunk_size:
            return torch.sum(x, dim=dim)
        
        # 使用固定块大小进行分块求和
        chunks = torch.chunk(x, chunks=(x.size(dim) + fixed_chunk_size - 1) // fixed_chunk_size, dim=dim)
        result = torch.zeros_like(chunks[0].sum(dim=dim))
        
        for chunk in chunks:
            result += chunk.sum(dim=dim)
        
        return result
    
    def batch_invariant_mean(self, x: torch.Tensor, dim: int = -1, 
                            fixed_chunk_size: int = 64) -> torch.Tensor:
        """批处理不变性均值"""
        if x.size(dim) <= fixed_chunk_size:
            return torch.mean(x, dim=dim)
        
        # 使用固定块大小进行分块计算
        chunks = torch.chunk(x, chunks=(x.size(dim) + fixed_chunk_size - 1) // fixed_chunk_size, dim=dim)
        total_sum = torch.zeros_like(chunks[0].sum(dim=dim))
        total_count = 0
        
        for chunk in chunks:
            total_sum += chunk.sum(dim=dim)
            total_count += chunk.size(dim)
        
        return total_sum / total_count
    
    def batch_invariant_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 num_heads: int = 8, fixed_kv_chunk_size: int = 64) -> torch.Tensor:
        """批处理不变性注意力机制"""
        batch_size, seq_len, hidden_dim = q.shape
        head_dim = hidden_dim // num_heads
        
        # 重塑为多头格式
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        # 使用固定块大小处理Value
        if seq_len <= fixed_kv_chunk_size:
            out = torch.matmul(attn_weights, v)
        else:
            # 分块处理 - 对每个块的结果进行加权平均
            num_chunks = (seq_len + fixed_kv_chunk_size - 1) // fixed_kv_chunk_size
            outputs = []
            
            for i in range(num_chunks):
                start_idx = i * fixed_kv_chunk_size
                end_idx = min(start_idx + fixed_kv_chunk_size, seq_len)
                
                v_chunk = v[:, :, start_idx:end_idx, :]
                attn_chunk = attn_weights[:, :, :, start_idx:end_idx]
                
                out_chunk = torch.matmul(attn_chunk, v_chunk)
                outputs.append(out_chunk)
            
            # 对每个块的结果进行加权平均
            out = torch.zeros_like(outputs[0])
            for output in outputs:
                out += output / len(outputs)
        
        # 重塑回原始格式
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        
        return out
    
    def batch_invariant_layer_norm(self, x: torch.Tensor, normalized_shape: Tuple[int, ...],
                                  fixed_chunk_size: int = 64) -> torch.Tensor:
        """批处理不变性层归一化"""
        if x.size(-1) <= fixed_chunk_size:
            return F.layer_norm(x, normalized_shape)
        
        # 使用固定块大小计算均值和方差
        chunks = torch.chunk(x, chunks=(x.size(-1) + fixed_chunk_size - 1) // fixed_chunk_size, dim=-1)
        
        # 计算总体均值和方差
        total_sum = torch.zeros_like(chunks[0].sum(dim=-1, keepdim=True))
        total_sum_sq = torch.zeros_like(chunks[0].sum(dim=-1, keepdim=True))
        total_count = 0
        
        for chunk in chunks:
            total_sum += chunk.sum(dim=-1, keepdim=True)
            total_sum_sq += (chunk ** 2).sum(dim=-1, keepdim=True)
            total_count += chunk.size(-1)
        
        mean = total_sum / total_count
        var = (total_sum_sq / total_count) - (mean ** 2)
        
        # 应用归一化
        x_normalized = (x - mean) / torch.sqrt(var + 1e-5)
        
        return x_normalized

class BatchInvariantDemo:
    """批处理不变性演示类"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.ops = BatchInvariantOps(device)
        self.results = {}
    
    def demonstrate_batch_invariance(self, seq_lengths: List[int] = [64, 128, 256, 512, 1024],
                                   hidden_dim: int = 512, num_heads: int = 8) -> Dict[str, List[float]]:
        """演示批处理不变性"""
        print("=== 批处理不变性演示 ===")
        
        results = {
            'standard_sum': [],
            'batch_invariant_sum': [],
            'standard_attention': [],
            'batch_invariant_attention': []
        }
        
        for seq_len in seq_lengths:
            print(f"测试序列长度: {seq_len}")
            
            # 创建输入数据
            x = torch.randn(2, seq_len, hidden_dim, device=self.device)
            q = torch.randn(2, seq_len, hidden_dim, device=self.device)
            k = torch.randn(2, seq_len, hidden_dim, device=self.device)
            v = torch.randn(2, seq_len, hidden_dim, device=self.device)
            
            # 测试标准求和
            torch.manual_seed(42)
            std_sum = torch.sum(x, dim=-1)
            results['standard_sum'].append(std_sum.detach().cpu().numpy())
            
            # 测试批处理不变性求和
            torch.manual_seed(42)
            bi_sum = self.ops.batch_invariant_sum(x, dim=-1, fixed_chunk_size=64)
            results['batch_invariant_sum'].append(bi_sum.detach().cpu().numpy())
            
            # 测试标准注意力
            torch.manual_seed(42)
            std_attn = self._standard_attention(q, k, v, num_heads)
            results['standard_attention'].append(std_attn.detach().cpu().numpy())
            
            # 测试批处理不变性注意力
            torch.manual_seed(42)
            bi_attn = self.ops.batch_invariant_attention(q, k, v, num_heads, fixed_kv_chunk_size=64)
            results['batch_invariant_attention'].append(bi_attn.detach().cpu().numpy())
        
        self.results = results
        return results
    
    def _standard_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                           num_heads: int) -> torch.Tensor:
        """标准注意力实现"""
        batch_size, seq_len, hidden_dim = q.shape
        head_dim = hidden_dim // num_heads
        
        # 重塑为多头格式
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        # 应用注意力权重
        out = torch.matmul(attn_weights, v)
        
        # 重塑回原始格式
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        
        return out
    
    def analyze_batch_invariance(self, results: Dict[str, List[np.ndarray]]) -> None:
        """分析批处理不变性结果"""
        print("\n=== 批处理不变性分析 ===")
        
        seq_lengths = [64, 128, 256, 512, 1024]
        
        for i, seq_len in enumerate(seq_lengths):
            print(f"序列长度 {seq_len}:")
            
            # 比较求和结果
            std_sum = results['standard_sum'][i]
            bi_sum = results['batch_invariant_sum'][i]
            sum_diff = np.mean(np.abs(std_sum - bi_sum))
            print(f"  求和差异: {sum_diff:.2e}")
            
            # 比较注意力结果
            std_attn = results['standard_attention'][i]
            bi_attn = results['batch_invariant_attention'][i]
            attn_diff = np.mean(np.abs(std_attn - bi_attn))
            print(f"  注意力差异: {attn_diff:.2e}")
            print()
    
    def visualize_batch_invariance(self, results: Dict[str, List[np.ndarray]], 
                                  save_path: str = None) -> None:
        """可视化批处理不变性结果"""
        seq_lengths = [64, 128, 256, 512, 1024]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 计算差异
        sum_diffs = []
        attn_diffs = []
        
        for i in range(len(seq_lengths)):
            std_sum = results['standard_sum'][i]
            bi_sum = results['batch_invariant_sum'][i]
            sum_diff = np.mean(np.abs(std_sum - bi_sum))
            sum_diffs.append(sum_diff)
            
            std_attn = results['standard_attention'][i]
            bi_attn = results['batch_invariant_attention'][i]
            attn_diff = np.mean(np.abs(std_attn - bi_attn))
            attn_diffs.append(attn_diff)
        
        # 子图1: 求和差异
        ax1 = axes[0, 0]
        ax1.plot(seq_lengths, sum_diffs, 'o-', linewidth=2, markersize=8, color='blue')
        ax1.set_title('求和操作差异', fontsize=14)
        ax1.set_xlabel('序列长度', fontsize=12)
        ax1.set_ylabel('平均绝对差异', fontsize=12)
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 注意力差异
        ax2 = axes[0, 1]
        ax2.plot(seq_lengths, attn_diffs, 'o-', linewidth=2, markersize=8, color='red')
        ax2.set_title('注意力机制差异', fontsize=14)
        ax2.set_xlabel('序列长度', fontsize=12)
        ax2.set_ylabel('平均绝对差异', fontsize=12)
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # 子图3: 差异对比
        ax3 = axes[1, 0]
        ax3.plot(seq_lengths, sum_diffs, 'o-', label='求和', linewidth=2, markersize=8)
        ax3.plot(seq_lengths, attn_diffs, 's-', label='注意力', linewidth=2, markersize=8)
        ax3.set_title('操作差异对比', fontsize=14)
        ax3.set_xlabel('序列长度', fontsize=12)
        ax3.set_ylabel('平均绝对差异', fontsize=12)
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 子图4: 相对差异
        ax4 = axes[1, 1]
        rel_sum_diffs = [diff / np.mean(np.abs(results['standard_sum'][i])) for i, diff in enumerate(sum_diffs)]
        rel_attn_diffs = [diff / np.mean(np.abs(results['standard_attention'][i])) for i, diff in enumerate(attn_diffs)]
        
        ax4.plot(seq_lengths, rel_sum_diffs, 'o-', label='求和', linewidth=2, markersize=8)
        ax4.plot(seq_lengths, rel_attn_diffs, 's-', label='注意力', linewidth=2, markersize=8)
        ax4.set_title('相对差异对比', fontsize=14)
        ax4.set_xlabel('序列长度', fontsize=12)
        ax4.set_ylabel('相对差异', fontsize=12)
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"批处理不变性图表已保存到: {save_path}")
        
        plt.show()
    
    def benchmark_batch_invariant_ops(self, seq_lengths: List[int] = [64, 128, 256, 512, 1024],
                                    hidden_dim: int = 512, num_trials: int = 100) -> Dict[str, List[float]]:
        """批处理不变性操作性能基准测试"""
        print("=== 批处理不变性操作性能基准测试 ===")
        
        performance_results = {
            'standard_sum': [],
            'batch_invariant_sum': [],
            'standard_attention': [],
            'batch_invariant_attention': []
        }
        
        for seq_len in seq_lengths:
            print(f"测试序列长度: {seq_len}")
            
            # 创建输入数据
            x = torch.randn(2, seq_len, hidden_dim, device=self.device)
            q = torch.randn(2, seq_len, hidden_dim, device=self.device)
            k = torch.randn(2, seq_len, hidden_dim, device=self.device)
            v = torch.randn(2, seq_len, hidden_dim, device=self.device)
            
            # 预热
            for _ in range(10):
                _ = torch.sum(x, dim=-1)
                _ = self.ops.batch_invariant_sum(x, dim=-1)
                _ = self._standard_attention(q, k, v, 8)
                _ = self.ops.batch_invariant_attention(q, k, v, 8)
            
            # 测试标准求和
            start_time = time.time()
            for _ in range(num_trials):
                _ = torch.sum(x, dim=-1)
            std_sum_time = (time.time() - start_time) / num_trials
            performance_results['standard_sum'].append(std_sum_time)
            
            # 测试批处理不变性求和
            start_time = time.time()
            for _ in range(num_trials):
                _ = self.ops.batch_invariant_sum(x, dim=-1)
            bi_sum_time = (time.time() - start_time) / num_trials
            performance_results['batch_invariant_sum'].append(bi_sum_time)
            
            # 测试标准注意力
            start_time = time.time()
            for _ in range(num_trials):
                _ = self._standard_attention(q, k, v, 8)
            std_attn_time = (time.time() - start_time) / num_trials
            performance_results['standard_attention'].append(std_attn_time)
            
            # 测试批处理不变性注意力
            start_time = time.time()
            for _ in range(num_trials):
                _ = self.ops.batch_invariant_attention(q, k, v, 8)
            bi_attn_time = (time.time() - start_time) / num_trials
            performance_results['batch_invariant_attention'].append(bi_attn_time)
            
            print(f"  标准求和: {std_sum_time*1000:.2f}ms")
            print(f"  批处理不变性求和: {bi_sum_time*1000:.2f}ms")
            print(f"  标准注意力: {std_attn_time*1000:.2f}ms")
            print(f"  批处理不变性注意力: {bi_attn_time*1000:.2f}ms")
            print(f"  求和开销: {(bi_sum_time/std_sum_time-1)*100:.1f}%")
            print(f"  注意力开销: {(bi_attn_time/std_attn_time-1)*100:.1f}%")
        
        return performance_results
    
    def visualize_performance(self, performance_results: Dict[str, List[float]], 
                            seq_lengths: List[int], save_path: str = None) -> None:
        """可视化性能结果"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 子图1: 求和性能
        ax1 = axes[0]
        ax1.plot(seq_lengths, [t*1000 for t in performance_results['standard_sum']], 
                'o-', label='标准求和', linewidth=2, markersize=8)
        ax1.plot(seq_lengths, [t*1000 for t in performance_results['batch_invariant_sum']], 
                's-', label='批处理不变性求和', linewidth=2, markersize=8)
        ax1.set_title('求和操作性能对比', fontsize=14)
        ax1.set_xlabel('序列长度', fontsize=12)
        ax1.set_ylabel('平均执行时间 (ms)', fontsize=12)
        ax1.legend()
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 注意力性能
        ax2 = axes[1]
        ax2.plot(seq_lengths, [t*1000 for t in performance_results['standard_attention']], 
                'o-', label='标准注意力', linewidth=2, markersize=8)
        ax2.plot(seq_lengths, [t*1000 for t in performance_results['batch_invariant_attention']], 
                's-', label='批处理不变性注意力', linewidth=2, markersize=8)
        ax2.set_title('注意力机制性能对比', fontsize=14)
        ax2.set_xlabel('序列长度', fontsize=12)
        ax2.set_ylabel('平均执行时间 (ms)', fontsize=12)
        ax2.legend()
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"性能对比图表已保存到: {save_path}")
        
        plt.show()
    
    def run_complete_demo(self) -> None:
        """运行完整的批处理不变性演示"""
        print("开始批处理不变性完整演示...\n")
        
        # 1. 演示批处理不变性
        results = self.demonstrate_batch_invariance()
        
        # 2. 分析结果
        self.analyze_batch_invariance(results)
        
        # 3. 可视化结果
        self.visualize_batch_invariance(results, 'experiments/plots/batch_invariance.png')
        
        # 4. 性能基准测试
        seq_lengths = [64, 128, 256, 512, 1024]
        performance_results = self.benchmark_batch_invariant_ops(seq_lengths)
        
        # 5. 可视化性能
        self.visualize_performance(performance_results, seq_lengths, 
                                 'experiments/plots/batch_invariant_performance.png')
        
        print("\n批处理不变性演示完成！")

if __name__ == "__main__":
    demo = BatchInvariantDemo()
    demo.run_complete_demo()
