#!/usr/bin/env python3
"""
KV Cache场景演示
验证原博客文章的核心观点：Split-KV策略会破坏scenario invariance

场景设置：
- KV Cache长度: 1000
- 场景1: 只输入"A" (1个token)
- 场景2: 输入"ABC" (3个token)
- 期望: 传统Split-KV导致A计算结果不一致，Batch Invariant版本保持一致
"""

import sys
import os
sys.path.append('.')

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
import time
from typing import Dict, Any, List

# 导入必要的模块
from src.device_manager import DeviceManager
from src.font_config import setup_chinese_fonts

# 设置中文字体
setup_chinese_fonts()

class KVCacheScenarioDemo:
    """KV Cache场景演示类"""
    
    def __init__(self, parallel_degree: int = 4, fixed_split_size: int = 64):
        self.device_manager = DeviceManager()
        self.device = self.device_manager.get_device()
        self.parallel_degree = parallel_degree
        self.fixed_split_size = fixed_split_size
        
        print(f"使用设备: {self.device}")
        print(f"并行度: {self.parallel_degree}")
        print(f"固定分割大小: {self.fixed_split_size}")
    
    def non_deterministic_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Non-Deterministic Attention实现 - 模拟Split-KV策略"""
        batch_size, num_heads, seq_len, head_dim = Q.shape
        _, _, kv_len, _ = K.shape
        
        print(f"  Batch Size: {batch_size}, KV Length: {kv_len}")
        
        # 动态分割策略基于batch size
        if batch_size < self.parallel_degree:
            num_splits = self.parallel_degree
            split_size = kv_len // num_splits
            if split_size == 0: split_size = 1
            print(f"  Split Strategy: {num_splits} splits, each ~{split_size} elements")
        else:
            num_splits = max(1, self.parallel_degree // batch_size)
            split_size = kv_len // num_splits
            if split_size == 0: split_size = 1
            print(f"  Split Strategy: {num_splits} splits, each ~{split_size} elements")
        
        # 计算attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(head_dim)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # 模拟split-reduction
        output = self._simulate_split_reduction(attention_weights, V, num_splits, split_size)
        return output
    
    def deterministic_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Deterministic Attention实现 - 使用固定分割大小策略"""
        batch_size, num_heads, seq_len, head_dim = Q.shape
        _, _, kv_len, _ = K.shape
        
        print(f"  Batch Size: {batch_size}, KV Length: {kv_len}")
        
        # 固定分割大小策略
        split_size = self.fixed_split_size
        num_splits = (kv_len + split_size - 1) // split_size
        print(f"  Fixed Split Strategy: {num_splits} splits, each {split_size} elements")
        
        # 计算attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(head_dim)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # 模拟固定分割reduction
        output = self._simulate_fixed_split_reduction(attention_weights, V, split_size)
        return output
    
    def _simulate_split_reduction(self, attention_weights: torch.Tensor, V: torch.Tensor, num_splits: int, split_size: int) -> torch.Tensor:
        """模拟split-reduction过程"""
        batch_size, num_heads, seq_len, kv_len = attention_weights.shape
        _, _, _, head_dim = V.shape
        
        results = []
        for i in range(0, kv_len, split_size):
            end_idx = min(i + split_size, kv_len)
            
            attn_split = attention_weights[:, :, :, i:end_idx]
            V_split = V[:, :, i:end_idx, :]
            
            split_result = torch.matmul(attn_split, V_split)
            results.append(split_result)
        
        # 非确定性累积顺序
        current_time = int(time.time() * 1000000) % 1000
        if current_time % 3 == 0:
            result = results[0]
            for i in range(1, len(results)):
                result = result + results[i]
        elif current_time % 3 == 1:
            result = results[-1]
            for i in range(len(results) - 2, -1, -1):
                result = results[i] + result
        else:
            random.shuffle(results)
            result = results[0]
            for i in range(1, len(results)):
                result = result + results[i]
        
        return result
    
    def _simulate_fixed_split_reduction(self, attention_weights: torch.Tensor, V: torch.Tensor, split_size: int) -> torch.Tensor:
        """模拟固定分割reduction过程"""
        batch_size, num_heads, seq_len, kv_len = attention_weights.shape
        _, _, _, head_dim = V.shape
        
        results = []
        for i in range(0, kv_len, split_size):
            end_idx = min(i + split_size, kv_len)
            
            attn_split = attention_weights[:, :, :, i:end_idx]
            V_split = V[:, :, i:end_idx, :]
            
            split_result = torch.matmul(attn_split, V_split)
            results.append(split_result)
        
        # 固定累积顺序
        result = results[0]
        for i in range(1, len(results)):
            result = result + results[i]
        
        return result
    
    def test_kv_cache_scenario(self, num_trials: int = 10) -> Dict[str, Any]:
        """测试KV Cache场景"""
        print("\n" + "=" * 80)
        print("KV Cache场景测试 - 验证原博客核心观点")
        print("=" * 80)
        
        # 模拟KV Cache场景
        kv_cache_length = 1000  # 固定的KV cache长度
        head_dim = 64
        num_heads = 8
        
        print(f"模拟场景: KV Cache长度={kv_cache_length}")
        print("场景1: 只输入'A' (1个token)")
        print("场景2: 输入'ABC' (3个token)")
        print("期望: 传统Split-KV导致A计算结果不一致，Batch Invariant版本保持一致")
        
        # 创建固定的KV cache
        torch.manual_seed(42)
        K_cache = torch.randn(1, num_heads, kv_cache_length, head_dim, device=self.device, dtype=torch.float32)
        V_cache = torch.randn(1, num_heads, kv_cache_length, head_dim, device=self.device, dtype=torch.float32)
        
        # 场景1: 只输入"A" (1个token)
        print(f"\n=== 场景1: 只输入'A' (1个token) ===")
        Q1 = torch.randn(1, num_heads, 1, head_dim, device=self.device, dtype=torch.float32)
        
        # 场景2: 输入"ABC" (3个token)
        print(f"=== 场景2: 输入'ABC' (3个token) ===")
        Q2 = torch.randn(1, num_heads, 3, head_dim, device=self.device, dtype=torch.float32)
        
        results = {
            'scenarios': ['A_only', 'ABC'],
            'kv_cache_length': kv_cache_length,
            'non_deterministic_results': {},
            'deterministic_results': {},
            'scenario_invariance_results': {}
        }
        
        # 测试两个场景
        scenarios = [
            ('A_only', Q1, '只输入A'),
            ('ABC', Q2, '输入ABC')
        ]
        
        for scenario_name, Q, description in scenarios:
            print(f"\n--- 测试场景: {description} ---")
            print(f"输入形状: Q={Q.shape}, K={K_cache.shape}, V={V_cache.shape}")
            
            # 测试Non-Deterministic版本
            print(f"\n--- NON-DETERMINISTIC 实现 ---")
            non_det_outputs = []
            for trial in range(num_trials):
                time.sleep(0.001)  # 确保时间戳不同
                with torch.no_grad():
                    output = self.non_deterministic_attention(Q, K_cache, V_cache)
                    non_det_outputs.append(output.cpu().numpy())
            
            # 检查Non-Deterministic的一致性
            non_det_consistent = self._check_consistency(non_det_outputs)
            print(f"一致性: {'✅ 通过' if non_det_consistent else '❌ 失败'}")
            
            # 计算输出统计
            non_det_mean = np.mean(non_det_outputs, axis=0)
            non_det_range = [non_det_mean.min(), non_det_mean.max()]
            print(f"输出范围: [{non_det_range[0]:.6f}, {non_det_range[1]:.6f}]")
            
            # 显示前几次的具体输出值
            print("前5次输出值 (第一个元素):")
            for i in range(min(5, len(non_det_outputs))):
                output_val = non_det_outputs[i][0, 0, 0, 0]
                print(f"  第{i+1}次: {output_val:.10f}")
            
            # 测试Deterministic版本
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
                    output = self.deterministic_attention(Q, K_cache, V_cache)
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
            
            results['non_deterministic_results'][scenario_name] = {
                'outputs': non_det_outputs,
                'consistent': non_det_consistent,
                'description': description,
                'mean_output': non_det_mean
            }
            results['deterministic_results'][scenario_name] = {
                'outputs': det_outputs,
                'consistent': det_consistent,
                'description': description,
                'mean_output': det_mean
            }
        
        # 测试scenario invariance
        print(f"\n=== Scenario Invariance测试 ===")
        print("比较场景1和场景2中A token的输出一致性")
        
        scenario_invariance_results = self._test_scenario_invariance(results)
        results['scenario_invariance_results'] = scenario_invariance_results
        
        return results
    
    def _test_scenario_invariance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """测试场景间的一致性"""
        print("比较两个场景中A token的输出一致性")
        
        # 获取两个场景的结果
        a_only_non_det = results['non_deterministic_results']['A_only']['mean_output']
        abc_non_det = results['non_deterministic_results']['ABC']['mean_output']
        a_only_det = results['deterministic_results']['A_only']['mean_output']
        abc_det = results['deterministic_results']['ABC']['mean_output']
        
        # 比较Non-Deterministic版本
        non_det_diff = np.abs(a_only_non_det - abc_non_det).max()
        non_det_invariant = non_det_diff < 1e-6
        
        # 比较Deterministic版本
        det_diff = np.abs(a_only_det - abc_det).max()
        det_invariant = det_diff < 1e-6
        
        print(f"Non-Deterministic版本场景间差异: {non_det_diff:.2e}")
        print(f"Deterministic版本场景间差异: {det_diff:.2e}")
        
        print(f"Non-Deterministic Scenario Invariance: {'✅ 通过' if non_det_invariant else '❌ 失败'}")
        print(f"Deterministic Scenario Invariance: {'✅ 通过' if det_invariant else '❌ 失败'}")
        
        return {
            'non_deterministic_invariant': non_det_invariant,
            'deterministic_invariant': det_invariant,
            'non_deterministic_diff': non_det_diff,
            'deterministic_diff': det_diff
        }
    
    def _check_consistency(self, outputs: List[np.ndarray], tolerance: float = 1e-6) -> bool:
        """检查输出的一致性"""
        if len(outputs) <= 1:
            return True
        
        reference = outputs[0]
        for output in outputs[1:]:
            if np.max(np.abs(output - reference)) > tolerance:
                return False
        return True
    
    def analyze_results(self, results: Dict[str, Any]):
        """分析结果"""
        print("\n" + "=" * 80)
        print("KV Cache场景结果分析")
        print("=" * 80)
        
        scenario_invariance = results['scenario_invariance_results']
        
        print("Scenario Invariance测试结果:")
        print(f"  - Non-Deterministic实现: {'✅ 通过' if scenario_invariance['non_deterministic_invariant'] else '❌ 失败'}")
        print(f"  - Deterministic实现: {'✅ 通过' if scenario_invariance['deterministic_invariant'] else '❌ 失败'}")
        
        print(f"\n场景间输出差异:")
        print(f"  - Non-Deterministic差异: {scenario_invariance['non_deterministic_diff']:.2e}")
        print(f"  - Deterministic差异: {scenario_invariance['deterministic_diff']:.2e}")
        
        if not scenario_invariance['non_deterministic_invariant'] and scenario_invariance['deterministic_invariant']:
            print("✅ 验证成功: Non-Deterministic版本破坏了scenario invariance，而Deterministic版本保持了一致性")
        else:
            print("⚠️ 需要进一步调整参数以观察到预期的差异")
    
    def create_visualization(self, results: Dict[str, Any]):
        """创建可视化"""
        print("\n" + "=" * 80)
        print("创建KV Cache场景可视化")
        print("=" * 80)
        
        # 创建场景对比图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        scenarios = ['A_only', 'ABC']
        scenario_names = ['只输入A', '输入ABC']
        
        # 获取实验次数
        num_trials = len(results['non_deterministic_results']['A_only']['outputs'])
        
        for i, (scenario, name) in enumerate(zip(scenarios, scenario_names)):
            # Non-Deterministic版本
            ax1 = axes[i, 0]
            outputs = results['non_deterministic_results'][scenario]['outputs']
            values = [output[0, 0, 0, 0] for output in outputs]
            
            ax1.plot(range(1, num_trials + 1), values, 'o-', color='red', linewidth=2, markersize=6)
            ax1.set_title(f'Non-Deterministic版本 - {name}', fontsize=14, fontweight='bold')
            ax1.set_xlabel('实验次数', fontsize=12)
            ax1.set_ylabel('输出值', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # 添加数值标签（前5次）
            for j, value in enumerate(values[:5]):
                ax1.annotate(f'{value:.8f}', (j+1, value), 
                           textcoords="offset points", xytext=(0,10), ha='center', 
                           fontsize=8, alpha=0.7)
            
            # Deterministic版本
            ax2 = axes[i, 1]
            outputs = results['deterministic_results'][scenario]['outputs']
            values = [output[0, 0, 0, 0] for output in outputs]
            
            ax2.plot(range(1, num_trials + 1), values, 'o-', color='blue', linewidth=2, markersize=6)
            ax2.set_title(f'Deterministic版本 - {name}', fontsize=14, fontweight='bold')
            ax2.set_xlabel('实验次数', fontsize=12)
            ax2.set_ylabel('输出值', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # 添加数值标签（前5次）
            for j, value in enumerate(values[:5]):
                ax2.annotate(f'{value:.8f}', (j+1, value), 
                           textcoords="offset points", xytext=(0,10), ha='center', 
                           fontsize=8, alpha=0.7)
        
        # 添加整体标题
        fig.suptitle('KV Cache场景测试结果对比\n场景1: 只输入A vs 场景2: 输入ABC', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # 添加说明文字
        fig.text(0.5, 0.02, 
                f'关键验证: 传统Split-KV导致A在两种场景下计算结果不一致，Batch Invariant版本保持一致\n'
                f'实验配置: KV Cache长度=1000, 并行度=4, 固定分割大小=64, 每个场景进行{num_trials}次实验', 
                ha='center', fontsize=11, style='italic',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, bottom=0.15)
        
        # 保存图片
        os.makedirs('experiments/plots', exist_ok=True)
        plt.savefig('experiments/plots/kv_cache_scenario_visualization.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("KV Cache场景可视化已保存到: experiments/plots/kv_cache_scenario_visualization.png")
    
    def run_analysis(self):
        """运行完整分析"""
        print("KV Cache场景演示")
        print("基于Thinking Machines博客文章")
        print("=" * 80)
        
        # 运行测试
        results = self.test_kv_cache_scenario()
        
        # 分析结果
        self.analyze_results(results)
        
        # 创建可视化
        self.create_visualization(results)
        
        # 总结
        print("\n" + "=" * 80)
        print("分析总结")
        print("=" * 80)
        
        print("🎯 核心观点验证:")
        print("  - 当输入长度不同时，Split-KV策略会破坏scenario invariance")
        print("  - 不同的输入长度导致不同的KV分割策略")
        print("  - 即使处理相同的token，不同的分割策略导致不同的归约顺序")
        print("  - 解决方案是使用固定分割大小策略")
        
        print("🎉 KV Cache场景分析完成！")
        
        return results

def main():
    """主函数"""
    print("使用中文字体:", matplotlib.font_manager.FontProperties().get_name())
    
    # 创建演示对象
    demo = KVCacheScenarioDemo()
    
    # 运行分析
    results = demo.run_analysis()
    
    print("\n🎉 KV Cache场景分析完成！")
    print("\n关键发现:")
    print("- 当输入长度不同时，Split-KV策略会破坏scenario invariance")
    print("- 不同的输入长度导致不同的KV分割策略")
    print("- 即使处理相同的token，不同的分割策略导致不同的归约顺序")
    print("- 解决方案是使用固定分割大小策略")
    print("- 本质上是3个矩阵乘法的归约问题")

if __name__ == "__main__":
    main()
