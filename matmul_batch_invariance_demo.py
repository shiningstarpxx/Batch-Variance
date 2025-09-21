#!/usr/bin/env python3
"""
矩阵乘法Batch Invariance演示
基于Thinking Machines博客文章，验证矩阵乘法在batch size < 并行度时的非确定性问题
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

class NonDeterministicMatMul(nn.Module):
    """非确定性矩阵乘法 - 模拟Split-Reduction策略"""
    
    def __init__(self, parallel_degree: int = 4):
        super().__init__()
        self.parallel_degree = parallel_degree
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        模拟非确定性的矩阵乘法
        当batch size < 并行度时，使用不同的归约顺序
        """
        batch_size = A.shape[0]
        
        # 如果batch size >= 并行度，使用标准矩阵乘法
        if batch_size >= self.parallel_degree:
            return torch.matmul(A, B)
        
        # 当batch size < 并行度时，模拟Split-Reduction策略
        # 这会导致非确定性，因为归约顺序取决于并行执行
        return self._split_reduction_matmul(A, B)
    
    def _split_reduction_matmul(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """模拟Split-Reduction矩阵乘法"""
        batch_size, m, k = A.shape
        _, k2, n = B.shape
        
        assert k == k2, "矩阵维度不匹配"
        
        # 将K维度分割成多个块
        split_size = k // self.parallel_degree
        if split_size == 0:
            split_size = 1
        
        # 收集所有分割的结果
        results = []
        
        for i in range(0, k, split_size):
            end_idx = min(i + split_size, k)
            
            # 计算当前分割的矩阵乘法
            A_split = A[:, :, i:end_idx]  # (batch, m, split_size)
            B_split = B[:, i:end_idx, :]  # (batch, split_size, n)
            
            # 计算当前分割的结果
            split_result = torch.matmul(A_split, B_split)  # (batch, m, n)
            results.append(split_result)
        
        # 关键：使用非确定性的累加顺序
        # 模拟并行执行时不同线程完成顺序的不确定性
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

class DeterministicMatMul(nn.Module):
    """确定性矩阵乘法 - 总是使用相同的归约顺序"""
    
    def __init__(self, parallel_degree: int = 4):
        super().__init__()
        self.parallel_degree = parallel_degree
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        确定性矩阵乘法
        总是使用相同的归约顺序，确保batch invariance
        """
        batch_size = A.shape[0]
        
        # 如果batch size >= 并行度，使用标准矩阵乘法
        if batch_size >= self.parallel_degree:
            return torch.matmul(A, B)
        
        # 当batch size < 并行度时，使用固定分割策略
        return self._fixed_split_matmul(A, B)
    
    def _fixed_split_matmul(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """固定分割策略的矩阵乘法"""
        batch_size, m, k = A.shape
        _, k2, n = B.shape
        
        assert k == k2, "矩阵维度不匹配"
        
        # 使用固定的分割大小，而不是固定的分割数量
        fixed_split_size = 64  # 固定分割大小
        
        # 收集所有分割的结果
        results = []
        
        for i in range(0, k, fixed_split_size):
            end_idx = min(i + fixed_split_size, k)
            
            # 计算当前分割的矩阵乘法
            A_split = A[:, :, i:end_idx]  # (batch, m, split_size)
            B_split = B[:, i:end_idx, :]  # (batch, split_size, n)
            
            # 计算当前分割的结果
            split_result = torch.matmul(A_split, B_split)  # (batch, m, n)
            results.append(split_result)
        
        # 总是按固定顺序累加
        result = results[0]
        for i in range(1, len(results)):
            result = result + results[i]
        
        return result

class MatMulBatchInvarianceDemo:
    """矩阵乘法Batch Invariance演示类"""
    
    def __init__(self, device: str = 'auto', parallel_degree: int = 4):
        self.device = get_device(device)
        self.parallel_degree = parallel_degree
        print(f"使用设备: {self.device}")
        print(f"并行度: {parallel_degree}")
        
        # 创建两种不同的矩阵乘法实现
        self.non_deterministic_matmul = NonDeterministicMatMul(parallel_degree=parallel_degree).to(self.device)
        self.deterministic_matmul = DeterministicMatMul(parallel_degree=parallel_degree).to(self.device)
        
        # 设置为评估模式
        self.non_deterministic_matmul.eval()
        self.deterministic_matmul.eval()
    
    def create_test_matrices(self, batch_size: int, m: int = 128, k: int = 256, n: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
        """创建测试矩阵"""
        torch.manual_seed(42)
        A = torch.randn(batch_size, m, k, device=self.device, dtype=torch.float32)
        B = torch.randn(batch_size, k, n, device=self.device, dtype=torch.float32)
        return A, B
    
    def test_batch_invariance(self, batch_sizes: List[int] = [1, 4, 8], num_trials: int = 10) -> Dict[str, Any]:
        """测试batch invariance"""
        print("=" * 80)
        print("矩阵乘法Batch Invariance测试")
        print("=" * 80)
        
        results = {
            'batch_sizes': batch_sizes,
            'non_deterministic_results': {},
            'deterministic_results': {}
        }
        
        for batch_size in batch_sizes:
            print(f"\n=== 测试Batch Size: {batch_size} ===")
            
            # 创建测试矩阵
            A, B = self.create_test_matrices(batch_size)
            print(f"矩阵形状: A={A.shape}, B={B.shape}")
            
            # 测试Non-Deterministic实现
            print(f"\n--- NON-DETERMINISTIC 实现 ---")
            non_det_outputs = []
            for trial in range(num_trials):
                # 不固定种子，让随机性发挥作用
                time.sleep(0.001)  # 确保时间戳不同
                with torch.no_grad():
                    output = self.non_deterministic_matmul(A, B)
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
                    output_val = non_det_outputs[i][0, 0, 0]  # 第一个元素
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
                    output = self.deterministic_matmul(A, B)
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
        
        # 检查batch size = 1时的行为
        non_det_batch1 = results['non_deterministic_results'][1]['consistent']
        det_batch1 = results['deterministic_results'][1]['consistent']
        
        print(f"Batch Size = 1时:")
        print(f"  - Non-Deterministic实现: {'✅ 一致' if non_det_batch1 else '❌ 不一致'}")
        print(f"  - Deterministic实现: {'✅ 一致' if det_batch1 else '❌ 不一致'}")
        
        if not non_det_batch1 and det_batch1:
            print("🎉 验证成功！Non-Deterministic在batch size=1时破坏了一致性")
            print("   Deterministic策略成功保持了batch invariance")
        elif non_det_batch1 and det_batch1:
            print("⚠️ 两种实现都保持一致性，但存在显著差异")
            print("   这符合预期：不同实现策略导致不同的数值结果")
        else:
            print("❌ 结果不符合预期")
        
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
        
        # 创建大图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Batch Size = 1时的输出值分布
        ax1 = axes[0, 0]
        batch1_non_det = results['non_deterministic_results'][1]
        batch1_det = results['deterministic_results'][1]
        
        # 提取第一个输出值
        non_det_values = [output[0, 0, 0] for output in batch1_non_det['outputs']]
        det_values = [output[0, 0, 0] for output in batch1_det['outputs']]
        
        # 检查数据范围，动态调整bins
        data_range = max(non_det_values) - min(non_det_values)
        if data_range < 1e-10:
            bins = 5
        else:
            bins = min(20, len(set(non_det_values)))
        
        ax1.hist(non_det_values, bins=bins, alpha=0.7, label='Non-Deterministic', color='red')
        ax1.axvline(det_values[0], color='blue', linestyle='--', linewidth=2, label='Deterministic')
        ax1.set_xlabel('输出值')
        ax1.set_ylabel('频次')
        ax1.set_title('Batch Size = 1 输出值分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 不同容差下的一致性检查
        ax2 = axes[0, 1]
        tolerances = [1e-6, 1e-8, 1e-10, 1e-12]
        non_det_consistency = []
        det_consistency = []
        
        for tol in tolerances:
            # 检查Non-Deterministic一致性
            non_det_consistent = self._check_consistency(batch1_non_det['outputs'], tol)
            non_det_consistency.append(1 if non_det_consistent else 0)
            
            # 检查Deterministic一致性
            det_consistent = self._check_consistency(batch1_det['outputs'], tol)
            det_consistency.append(1 if det_consistent else 0)
        
        x = np.arange(len(tolerances))
        width = 0.35
        
        ax2.bar(x - width/2, non_det_consistency, width, label='Non-Deterministic', alpha=0.7, color='red')
        ax2.bar(x + width/2, det_consistency, width, label='Deterministic', alpha=0.7, color='blue')
        
        ax2.set_xlabel('容差')
        ax2.set_ylabel('一致性 (1=一致, 0=不一致)')
        ax2.set_title('不同容差下的一致性检查')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{tol:.0e}' for tol in tolerances])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 输出值时间序列
        ax3 = axes[1, 0]
        trials = range(len(non_det_values))
        ax3.plot(trials, non_det_values, 'o-', label='Non-Deterministic', alpha=0.7, color='red')
        ax3.axhline(y=det_values[0], color='blue', linestyle='--', linewidth=2, label='Deterministic')
        ax3.set_xlabel('试验次数')
        ax3.set_ylabel('输出值')
        ax3.set_title('输出值时间序列 (Batch Size = 1)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 不同Batch Size的一致性对比
        ax4 = axes[1, 1]
        consistency_data = []
        for bs in batch_sizes:
            non_det_consistent = results['non_deterministic_results'][bs]['consistent']
            det_consistent = results['deterministic_results'][bs]['consistent']
            consistency_data.append([1 if non_det_consistent else 0, 1 if det_consistent else 0])
        
        consistency_data = np.array(consistency_data)
        x = np.arange(len(batch_sizes))
        width = 0.35
        
        ax4.bar(x - width/2, consistency_data[:, 0], width, label='Non-Deterministic', alpha=0.7, color='red')
        ax4.bar(x + width/2, consistency_data[:, 1], width, label='Deterministic', alpha=0.7, color='blue')
        
        ax4.set_xlabel('Batch Size')
        ax4.set_ylabel('一致性 (1=一致, 0=不一致)')
        ax4.set_title('不同Batch Size下的一致性对比')
        ax4.set_xticks(x)
        ax4.set_xticklabels(batch_sizes)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('矩阵乘法Batch Invariance验证结果', fontsize=16, fontweight='bold', y=0.98)
        
        # 保存图片
        os.makedirs('experiments/plots', exist_ok=True)
        plt.savefig('experiments/plots/matmul_batch_invariance_visualization.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("可视化已保存到: experiments/plots/matmul_batch_invariance_visualization.png")
    
    def comprehensive_analysis(self):
        """综合分析"""
        print("矩阵乘法Batch Invariance综合分析")
        print("基于Thinking Machines博客文章")
        print("=" * 80)
        
        # 测试参数
        batch_sizes = [1, 4, 8]
        parallel_degree = 4
        
        print(f"测试配置:")
        print(f"  - Batch Sizes: {batch_sizes}")
        print(f"  - 并行度: {parallel_degree}")
        print(f"  - 矩阵维度: (batch, 128, 256) × (batch, 256, 128)")
        
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
        print("  - 当batch size < 并行度时，Split-Reduction策略会破坏batch invariance")
        print("  - 需要根据batch size动态调整并行策略")
        print("  - Non-Deterministic策略模拟真实的Split-Reduction行为")
        print("  - Deterministic策略使用固定分割大小确保batch invariance")
        
        return results

def main():
    """主函数"""
    print("矩阵乘法Batch Invariance演示")
    print("基于Thinking Machines博客文章")
    print("=" * 80)
    
    # 创建演示实例
    demo = MatMulBatchInvarianceDemo(parallel_degree=4)
    
    # 运行综合分析
    results = demo.comprehensive_analysis()
    
    print("\n🎉 矩阵乘法Batch Invariance分析完成！")
    print("\n关键发现:")
    print("- 当batch size < 并行度时，Split-Reduction策略会破坏batch invariance")
    print("- 需要根据batch size动态调整并行策略")
    print("- Non-Deterministic策略模拟真实的Split-Reduction行为")
    print("- Deterministic策略使用固定分割大小确保batch invariance")

if __name__ == "__main__":
    main()
