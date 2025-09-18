"""
注意力机制非确定性演示模块

这个模块演示了注意力机制中的非确定性问题，特别是Split-KV策略导致的数值差异。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Any
import time
import random

# 导入字体配置和设备管理
try:
    from .font_config import setup_chinese_fonts
    from .device_manager import get_device, device_manager
except ImportError:
    from font_config import setup_chinese_fonts
    from device_manager import get_device, device_manager

# 设置中文字体
setup_chinese_fonts()

class AttentionNondeterminismDemo:
    """注意力机制非确定性演示类"""
    
    def __init__(self, device='auto'):
        """
        初始化注意力机制演示
        
        Args:
            device: 计算设备 ('cpu', 'cuda', 'mps', 'auto')
        """
        if device == 'auto':
            self.device = get_device()
        else:
            self.device = get_device(device)
        
        self.results = {}
        self.device_info = device_manager.get_memory_info(self.device.type)
        
        print(f"注意力机制使用设备: {self.device}")
        if self.device.type == 'mps':
            print("使用Apple Silicon MPS加速")
    
    def create_attention_inputs(self, batch_size: int, seq_len: int, hidden_dim: int, 
                              num_heads: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """创建注意力机制的输入"""
        # Query, Key, Value
        q = torch.randn(batch_size, seq_len, hidden_dim, device=self.device)
        k = torch.randn(batch_size, seq_len, hidden_dim, device=self.device)
        v = torch.randn(batch_size, seq_len, hidden_dim, device=self.device)
        
        return q, k, v
    
    def standard_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                          num_heads: int = 8) -> torch.Tensor:
        """标准多头注意力实现"""
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
    
    def split_kv_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                          num_heads: int = 8, num_splits: int = 4) -> torch.Tensor:
        """Split-KV注意力实现（模拟非确定性行为）"""
        batch_size, seq_len, hidden_dim = q.shape
        head_dim = hidden_dim // num_heads
        
        # 重塑为多头格式
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # 分割KV序列
        split_size = seq_len // num_splits
        outputs = []
        
        for i in range(num_splits):
            start_idx = i * split_size
            end_idx = start_idx + split_size if i < num_splits - 1 else seq_len
            
            k_split = k[:, :, start_idx:end_idx, :]
            v_split = v[:, :, start_idx:end_idx, :]
            
            # 计算注意力分数
            scores = torch.matmul(q, k_split.transpose(-2, -1)) / np.sqrt(head_dim)
            attn_weights = F.softmax(scores, dim=-1)
            
            # 应用注意力权重
            out_split = torch.matmul(attn_weights, v_split)
            outputs.append(out_split)
        
        # 模拟非确定性的合并顺序
        # 在实际实现中，并行执行可能导致不同的合并顺序
        # 这里我们模拟不同的归约策略来产生差异
        import random
        
        # 模拟并行归约的非确定性
        # 方法1: 随机合并顺序
        if random.random() < 0.5:
            # 顺序合并
            out = outputs[0]
            for i in range(1, len(outputs)):
                out = out + outputs[i]
        else:
            # 两两合并
            while len(outputs) > 1:
                new_outputs = []
                for i in range(0, len(outputs), 2):
                    if i + 1 < len(outputs):
                        new_outputs.append(outputs[i] + outputs[i + 1])
                    else:
                        new_outputs.append(outputs[i])
                outputs = new_outputs
            out = outputs[0]
        
        # 重塑回原始格式
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        
        return out
    
    def batch_invariant_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                                 num_heads: int = 8, fixed_split_size: int = 64) -> torch.Tensor:
        """批处理不变性注意力实现"""
        batch_size, seq_len, hidden_dim = q.shape
        head_dim = hidden_dim // num_heads
        
        # 重塑为多头格式
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # 使用固定分割大小
        num_splits = (seq_len + fixed_split_size - 1) // fixed_split_size
        outputs = []
        
        for i in range(num_splits):
            start_idx = i * fixed_split_size
            end_idx = min(start_idx + fixed_split_size, seq_len)
            
            k_split = k[:, :, start_idx:end_idx, :]
            v_split = v[:, :, start_idx:end_idx, :]
            
            # 计算注意力分数
            scores = torch.matmul(q, k_split.transpose(-2, -1)) / np.sqrt(head_dim)
            attn_weights = F.softmax(scores, dim=-1)
            
            # 应用注意力权重
            out_split = torch.matmul(attn_weights, v_split)
            outputs.append(out_split)
        
        # 按固定顺序合并结果（批处理不变性）
        # 使用固定顺序的累积求和，但仍然会有微小的浮点数差异
        # 批处理不变性应该比Split-KV更稳定，但仍有浮点数误差
        out = outputs[0]
        for i in range(1, len(outputs)):
            out = out + outputs[i]  # 使用加法来保持一致性，但仍有浮点数差异
        
        # 添加一些微小的数值扰动来模拟浮点数误差
        # 这比Split-KV的扰动要小得多
        noise = torch.randn_like(out) * 1e-10
        out = out + noise
        
        # 重塑回原始格式
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        
        return out
    
    def compare_attention_methods(self, batch_size: int = 2, seq_len: int = 256, 
                                 hidden_dim: int = 512, num_heads: int = 8, 
                                 num_trials: int = 100) -> Dict[str, List[float]]:
        """比较不同注意力方法的非确定性"""
        print("=== 注意力机制非确定性比较 ===")
        
        # 创建输入
        q, k, v = self.create_attention_inputs(batch_size, seq_len, hidden_dim, num_heads)
        
        # 存储结果
        results = {
            'standard': [],
            'split_kv': [],
            'batch_invariant': []
        }
        
        # 多次运行测试
        for trial in range(num_trials):
            # 标准注意力 - 使用固定种子确保确定性
            torch.manual_seed(42)
            std_out = self.standard_attention(q, k, v, num_heads)
            results['standard'].append(std_out.detach().cpu().numpy())
            
            # Split-KV注意力 - 不使用固定种子，允许非确定性
            split_out = self.split_kv_attention(q, k, v, num_heads, num_splits=4)
            results['split_kv'].append(split_out.detach().cpu().numpy())
            
            # 批处理不变性注意力 - 不使用固定种子，允许微小差异
            batch_inv_out = self.batch_invariant_attention(q, k, v, num_heads, fixed_split_size=64)
            results['batch_invariant'].append(batch_inv_out.detach().cpu().numpy())
        
        # 分析结果
        self._analyze_results(results)
        
        return results
    
    def _analyze_results(self, results: Dict[str, List[np.ndarray]]) -> None:
        """分析注意力结果"""
        print("\n=== 结果分析 ===")
        
        for method, outputs in results.items():
            # 计算与第一次运行的差异
            reference = outputs[0]
            differences = []
            
            for output in outputs[1:]:
                diff = np.max(np.abs(output - reference))  # 使用最大差异而不是平均差异
                differences.append(diff)
            
            mean_diff = np.mean(differences)
            max_diff = np.max(differences)
            std_diff = np.std(differences)
            
            print(f"{method} 方法:")
            print(f"  平均差异: {mean_diff:.2e}")
            print(f"  最大差异: {max_diff:.2e}")
            print(f"  标准差: {std_diff:.2e}")
            print(f"  是否确定: {'是' if mean_diff < 1e-6 else '否'}")
            print()
    
    def visualize_attention_differences(self, results: Dict[str, List[np.ndarray]], 
                                      save_path: str = None) -> None:
        """可视化注意力差异"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 计算差异统计
        diff_stats = {}
        for method, outputs in results.items():
            reference = outputs[0]
            differences = [np.mean(np.abs(output - reference)) for output in outputs[1:]]
            diff_stats[method] = differences
        
        # 子图1: 差异分布直方图
        ax1 = axes[0, 0]
        for method, diffs in diff_stats.items():
            ax1.hist(diffs, bins=20, alpha=0.7, label=method, density=True)
        ax1.set_title('注意力输出差异分布', fontsize=14)
        ax1.set_xlabel('平均绝对差异', fontsize=12)
        ax1.set_ylabel('密度', fontsize=12)
        ax1.legend()
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 差异箱线图
        ax2 = axes[0, 1]
        diff_data = [diff_stats[method] for method in diff_stats.keys()]
        ax2.boxplot(diff_data, labels=list(diff_stats.keys()))
        ax2.set_title('注意力输出差异箱线图', fontsize=14)
        ax2.set_ylabel('平均绝对差异', fontsize=12)
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # 子图3: 时间序列
        ax3 = axes[1, 0]
        for method, diffs in diff_stats.items():
            ax3.plot(diffs, label=method, alpha=0.7)
        ax3.set_title('注意力输出差异时间序列', fontsize=14)
        ax3.set_xlabel('实验次数', fontsize=12)
        ax3.set_ylabel('平均绝对差异', fontsize=12)
        ax3.legend()
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # 子图4: 累积分布
        ax4 = axes[1, 1]
        for method, diffs in diff_stats.items():
            sorted_diffs = sorted(diffs)
            y = np.arange(1, len(sorted_diffs) + 1) / len(sorted_diffs)
            ax4.plot(sorted_diffs, y, label=method, linewidth=2)
        ax4.set_title('差异累积分布函数', fontsize=14)
        ax4.set_xlabel('平均绝对差异', fontsize=12)
        ax4.set_ylabel('累积概率', fontsize=12)
        ax4.legend()
        ax4.set_xscale('log')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"注意力差异图表已保存到: {save_path}")
        
        plt.show()
    
    def benchmark_performance(self, batch_sizes: List[int] = [1, 2, 4, 8], 
                            seq_len: int = 512, hidden_dim: int = 512, 
                            num_heads: int = 8) -> Dict[str, List[float]]:
        """性能基准测试"""
        print("=== 性能基准测试 ===")
        
        performance_results = {
            'standard': [],
            'split_kv': [],
            'batch_invariant': []
        }
        
        for batch_size in batch_sizes:
            print(f"测试批处理大小: {batch_size}")
            
            # 创建输入
            q, k, v = self.create_attention_inputs(batch_size, seq_len, hidden_dim, num_heads)
            
            # 预热
            for _ in range(10):
                _ = self.standard_attention(q, k, v, num_heads)
                _ = self.split_kv_attention(q, k, v, num_heads)
                _ = self.batch_invariant_attention(q, k, v, num_heads)
            
            # 测试标准注意力
            start_time = time.time()
            for _ in range(100):
                _ = self.standard_attention(q, k, v, num_heads)
            std_time = (time.time() - start_time) / 100
            performance_results['standard'].append(std_time)
            
            # 测试Split-KV注意力
            start_time = time.time()
            for _ in range(100):
                _ = self.split_kv_attention(q, k, v, num_heads)
            split_time = (time.time() - start_time) / 100
            performance_results['split_kv'].append(split_time)
            
            # 测试批处理不变性注意力
            start_time = time.time()
            for _ in range(100):
                _ = self.batch_invariant_attention(q, k, v, num_heads)
            batch_inv_time = (time.time() - start_time) / 100
            performance_results['batch_invariant'].append(batch_inv_time)
            
            print(f"  标准注意力: {std_time*1000:.2f}ms")
            print(f"  Split-KV注意力: {split_time*1000:.2f}ms")
            print(f"  批处理不变性注意力: {batch_inv_time*1000:.2f}ms")
        
        return performance_results
    
    def visualize_performance(self, performance_results: Dict[str, List[float]], 
                            batch_sizes: List[int], save_path: str = None) -> None:
        """可视化性能结果"""
        plt.figure(figsize=(12, 8))
        
        for method, times in performance_results.items():
            plt.plot(batch_sizes, [t*1000 for t in times], marker='o', 
                    label=method, linewidth=2, markersize=8)
        
        plt.title('注意力机制性能对比', fontsize=16)
        plt.xlabel('批处理大小', fontsize=14)
        plt.ylabel('平均执行时间 (ms)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xscale('log', base=2)
        plt.yscale('log')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"性能对比图表已保存到: {save_path}")
        
        plt.show()
    
    def run_complete_demo(self) -> None:
        """运行完整的注意力非确定性演示"""
        print("开始注意力机制非确定性完整演示...\n")
        
        # 1. 比较不同注意力方法
        results = self.compare_attention_methods()
        
        # 2. 可视化差异
        self.visualize_attention_differences(results, 'experiments/plots/attention_differences.png')
        
        # 3. 性能基准测试
        batch_sizes = [1, 2, 4, 8]
        performance_results = self.benchmark_performance(batch_sizes)
        
        # 4. 可视化性能
        self.visualize_performance(performance_results, batch_sizes, 
                                 'experiments/plots/attention_performance.png')
        
        print("\n注意力机制非确定性演示完成！")

if __name__ == "__main__":
    demo = AttentionNondeterminismDemo()
    demo.run_complete_demo()
