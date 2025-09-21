#!/usr/bin/env python3
"""
分析为什么Batch-invariant方法仍然有差异

深入分析浮点数运算的细微差异来源
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

# 导入字体配置
try:
    from src.font_config import setup_chinese_fonts
except ImportError:
    from font_config import setup_chinese_fonts

setup_chinese_fonts()

class InvariantDifferenceAnalyzer:
    """分析Batch-invariant差异的类"""
    
    def __init__(self):
        """初始化分析器"""
        self.device = torch.device('cpu')  # 使用CPU确保精确性
    
    def analyze_floating_point_precision(self):
        """分析浮点数精度问题"""
        print("=== 浮点数精度分析 ===\n")
        
        # 创建测试数据
        torch.manual_seed(42)
        x = torch.randn(2, 4, 8, device=self.device)
        
        print("📊 测试数据:")
        print(f"输入形状: {x.shape}")
        print(f"输入数据类型: {x.dtype}")
        print(f"输入前几个值: {x[0, 0, :4]}")
        print()
        
        # 方法1: 标准RMSNorm (使用torch.mean)
        std_rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-6)
        std_output = x / std_rms
        
        # 方法2: 手动计算RMS (逐元素)
        manual_rms_squared = torch.zeros_like(std_rms)
        for i in range(x.shape[-1]):
            manual_rms_squared += x[:, :, i:i+1] ** 2
        manual_rms = torch.sqrt(manual_rms_squared / x.shape[-1] + 1e-6)
        manual_output = x / manual_rms
        
        # 方法3: 使用torch.sum然后除以长度
        sum_rms_squared = torch.sum(x ** 2, dim=-1, keepdim=True)
        sum_rms = torch.sqrt(sum_rms_squared / x.shape[-1] + 1e-6)
        sum_output = x / sum_rms
        
        # 计算差异
        std_manual_diff = torch.max(torch.abs(std_output - manual_output)).item()
        std_sum_diff = torch.max(torch.abs(std_output - sum_output)).item()
        manual_sum_diff = torch.max(torch.abs(manual_output - sum_output)).item()
        
        print("🔍 差异分析:")
        print(f"标准RMSNorm vs 手动计算: {std_manual_diff:.2e}")
        print(f"标准RMSNorm vs torch.sum: {std_sum_diff:.2e}")
        print(f"手动计算 vs torch.sum: {manual_sum_diff:.2e}")
        print()
        
        print("📈 详细数值对比 (第一个样本前4个值):")
        print(f"标准RMSNorm:     {std_output[0, 0, :4].numpy()}")
        print(f"手动计算:        {manual_output[0, 0, :4].numpy()}")
        print(f"torch.sum:       {sum_output[0, 0, :4].numpy()}")
        print()
        
        return {
            'std_output': std_output,
            'manual_output': manual_output,
            'sum_output': sum_output,
            'std_manual_diff': std_manual_diff,
            'std_sum_diff': std_sum_diff,
            'manual_sum_diff': manual_sum_diff
        }
    
    def analyze_accumulation_order(self):
        """分析累积顺序的影响"""
        print("=== 累积顺序影响分析 ===\n")
        
        # 创建测试数据
        torch.manual_seed(42)
        x = torch.randn(2, 4, 8, device=self.device)
        
        # 方法1: 顺序累积
        sequential_sum = torch.zeros(2, 4, 1, device=self.device)
        for i in range(x.shape[-1]):
            sequential_sum += x[:, :, i:i+1] ** 2
        
        # 方法2: 两两累积
        pairwise_sum = torch.zeros(2, 4, 1, device=self.device)
        for i in range(0, x.shape[-1], 2):
            if i + 1 < x.shape[-1]:
                pairwise_sum += x[:, :, i:i+1] ** 2 + x[:, :, i+1:i+2] ** 2
            else:
                pairwise_sum += x[:, :, i:i+1] ** 2
        
        # 方法3: 四四累积
        chunk_sum = torch.zeros(2, 4, 1, device=self.device)
        for i in range(0, x.shape[-1], 4):
            end_idx = min(i + 4, x.shape[-1])
            chunk_sum += torch.sum(x[:, :, i:end_idx] ** 2, dim=-1, keepdim=True)
        
        # 计算差异
        seq_pair_diff = torch.max(torch.abs(sequential_sum - pairwise_sum)).item()
        seq_chunk_diff = torch.max(torch.abs(sequential_sum - chunk_sum)).item()
        pair_chunk_diff = torch.max(torch.abs(pairwise_sum - chunk_sum)).item()
        
        print("🔍 累积顺序差异:")
        print(f"顺序 vs 两两: {seq_pair_diff:.2e}")
        print(f"顺序 vs 分块: {seq_chunk_diff:.2e}")
        print(f"两两 vs 分块: {pair_chunk_diff:.2e}")
        print()
        
        print("📈 累积结果对比:")
        print(f"顺序累积: {sequential_sum[0, 0, 0].item():.10f}")
        print(f"两两累积: {pairwise_sum[0, 0, 0].item():.10f}")
        print(f"分块累积: {chunk_sum[0, 0, 0].item():.10f}")
        print()
        
        return {
            'sequential_sum': sequential_sum,
            'pairwise_sum': pairwise_sum,
            'chunk_sum': chunk_sum,
            'seq_pair_diff': seq_pair_diff,
            'seq_chunk_diff': seq_chunk_diff,
            'pair_chunk_diff': pair_chunk_diff
        }
    
    def analyze_eps_effect(self):
        """分析eps参数的影响"""
        print("=== eps参数影响分析 ===\n")
        
        # 创建测试数据
        torch.manual_seed(42)
        x = torch.randn(2, 4, 8, device=self.device)
        
        eps_values = [1e-6, 1e-7, 1e-8, 1e-9, 0]
        results = {}
        
        for eps in eps_values:
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
            output = x / rms
            results[eps] = output
        
        print("🔍 eps参数对结果的影响:")
        base_output = results[1e-6]
        for eps, output in results.items():
            diff = torch.max(torch.abs(base_output - output)).item()
            print(f"eps={eps}: 差异 {diff:.2e}")
        print()
        
        return results
    
    def analyze_device_differences(self):
        """分析不同设备的差异"""
        print("=== 设备差异分析 ===\n")
        
        # 创建测试数据
        torch.manual_seed(42)
        x_cpu = torch.randn(2, 4, 8, device='cpu')
        
        # CPU结果
        cpu_rms = torch.sqrt(torch.mean(x_cpu ** 2, dim=-1, keepdim=True) + 1e-6)
        cpu_output = x_cpu / cpu_rms
        
        # MPS结果 (如果可用)
        if torch.backends.mps.is_available():
            x_mps = x_cpu.to('mps')
            mps_rms = torch.sqrt(torch.mean(x_mps ** 2, dim=-1, keepdim=True) + 1e-6)
            mps_output = x_mps / mps_rms
            mps_output_cpu = mps_output.cpu()
            
            diff = torch.max(torch.abs(cpu_output - mps_output_cpu)).item()
            print(f"CPU vs MPS差异: {diff:.2e}")
            
            print("📈 设备结果对比 (第一个样本前4个值):")
            print(f"CPU: {cpu_output[0, 0, :4].numpy()}")
            print(f"MPS: {mps_output_cpu[0, 0, :4].numpy()}")
        else:
            print("MPS不可用，跳过设备对比")
        
        print()
    
    def explain_why_differences_exist(self):
        """解释为什么存在差异"""
        print("=== 为什么Batch-invariant仍然有差异？ ===\n")
        
        print("🔍 主要原因:")
        print("1. **浮点数精度限制**: 即使使用相同的算法，浮点数运算仍有精度限制")
        print("2. **累积顺序差异**: 不同的累积顺序导致不同的舍入误差")
        print("3. **实现方式差异**: torch.mean vs 手动累积的实现方式不同")
        print("4. **设备差异**: CPU和GPU的浮点数运算可能有微小差异")
        print("5. **编译器优化**: 不同的优化级别可能产生不同的结果")
        print()
        
        print("💡 关键洞察:")
        print("• Batch-invariant的目标不是消除所有差异")
        print("• 而是确保相同输入在不同批处理大小下产生相同结果")
        print("• 微小的数值差异 (1e-6级别) 是可以接受的")
        print("• 重要的是消除非确定性的差异 (1e-3级别)")
        print()
        
        print("🎯 实际意义:")
        print("• 1e-6级别的差异对模型性能影响极小")
        print("• 但确保了推理的确定性和可重现性")
        print("• 这是科学计算中的常见现象")
        print("• 符合IEEE 754浮点数标准")
        print()
    
    def run_complete_analysis(self):
        """运行完整分析"""
        print("🚀 开始Batch-invariant差异分析...\n")
        
        # 1. 浮点数精度分析
        precision_results = self.analyze_floating_point_precision()
        
        # 2. 累积顺序分析
        accumulation_results = self.analyze_accumulation_order()
        
        # 3. eps参数分析
        eps_results = self.analyze_eps_effect()
        
        # 4. 设备差异分析
        self.analyze_device_differences()
        
        # 5. 解释差异原因
        self.explain_why_differences_exist()
        
        print("✅ Batch-invariant差异分析完成！")

def main():
    """主函数"""
    analyzer = InvariantDifferenceAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
