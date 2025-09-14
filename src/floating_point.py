"""
浮点数非结合性演示模块

这个模块演示了浮点数运算的非结合性特性，这是导致LLM推理非确定性的根本原因之一。
"""

import numpy as np
import torch
import random
from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# 导入字体配置
try:
    from .font_config import setup_chinese_fonts
except ImportError:
    from font_config import setup_chinese_fonts

# 设置中文字体
setup_chinese_fonts()

class FloatingPointDemo:
    """浮点数非结合性演示类"""
    
    def __init__(self):
        self.results = []
    
    def demonstrate_non_associativity(self) -> None:
        """演示浮点数的非结合性"""
        print("=== 浮点数非结合性演示 ===")
        
        # 基本示例
        a, b, c = 0.1, 1e20, -1e20
        result1 = (a + b) + c
        result2 = a + (b + c)
        
        print(f"a = {a}, b = {b}, c = {c}")
        print(f"(a + b) + c = {result1}")
        print(f"a + (b + c) = {result2}")
        print(f"两者相等吗？ {result1 == result2}")
        print(f"差异: {abs(result1 - result2)}")
        print()
    
    def demonstrate_sum_order_dependency(self, num_experiments: int = 10000) -> List[float]:
        """演示求和顺序依赖性"""
        print("=== 求和顺序依赖性演示 ===")
        
        # 创建包含不同数量级的数值
        vals = [1e-10, 1e-5, 1e-2, 1]
        vals = vals + [-v for v in vals]  # 添加负值，总和应该为0
        
        print(f"原始数组: {vals}")
        print(f"理论总和: {sum(vals)}")
        print()
        
        results = []
        random.seed(42)
        
        for _ in range(num_experiments):
            # 随机打乱数组顺序
            shuffled_vals = vals.copy()
            random.shuffle(shuffled_vals)
            result = sum(shuffled_vals)
            results.append(result)
        
        # 统计唯一结果
        unique_results = sorted(set(results))
        print(f"经过 {num_experiments} 次随机打乱后，得到 {len(unique_results)} 种不同的结果")
        print(f"结果范围: [{min(unique_results):.2e}, {max(unique_results):.2e}]")
        print(f"前10个唯一结果: {unique_results[:10]}")
        
        self.results = results
        return results
    
    def demonstrate_matrix_multiplication_determinism(self) -> None:
        """演示矩阵乘法的确定性"""
        print("=== 矩阵乘法确定性演示 ===")
        
        # 在Mac上使用CPU进行演示
        device = 'cpu'
        dtype = torch.float32
        
        # 创建随机矩阵
        A = torch.randn(512, 512, device=device, dtype=dtype)
        B = torch.randn(512, 512, device=device, dtype=dtype)
        
        # 计算参考结果
        ref = torch.mm(A, B)
        
        # 多次计算验证确定性
        num_tests = 100
        all_deterministic = True
        
        for i in range(num_tests):
            result = torch.mm(A, B)
            if not torch.allclose(result, ref, atol=1e-6):
                print(f"第 {i+1} 次计算与参考结果不同！")
                all_deterministic = False
                break
        
        if all_deterministic:
            print(f"经过 {num_tests} 次测试，矩阵乘法结果完全确定")
            print(f"最大差异: {torch.max(torch.abs(result - ref)).item():.2e}")
        else:
            print("发现非确定性结果！")
    
    def visualize_sum_distribution(self, save_path: str = None) -> None:
        """可视化求和结果分布"""
        if not self.results:
            print("请先运行 demonstrate_sum_order_dependency()")
            return
        
        plt.figure(figsize=(12, 8))
        
        # 子图1: 直方图
        plt.subplot(2, 2, 1)
        plt.hist(self.results, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('求和结果分布直方图', fontsize=14)
        plt.xlabel('求和结果', fontsize=12)
        plt.ylabel('频次', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 子图2: 箱线图
        plt.subplot(2, 2, 2)
        plt.boxplot(self.results, vert=True)
        plt.title('求和结果箱线图', fontsize=14)
        plt.ylabel('求和结果', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 子图3: 时间序列
        plt.subplot(2, 2, 3)
        plt.plot(self.results[:1000], alpha=0.7, color='green')
        plt.title('前1000次实验结果', fontsize=14)
        plt.xlabel('实验次数', fontsize=12)
        plt.ylabel('求和结果', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 子图4: 累积分布
        plt.subplot(2, 2, 4)
        sorted_results = sorted(self.results)
        y = np.arange(1, len(sorted_results) + 1) / len(sorted_results)
        plt.plot(sorted_results, y, color='red', linewidth=2)
        plt.title('累积分布函数', fontsize=14)
        plt.xlabel('求和结果', fontsize=12)
        plt.ylabel('累积概率', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()
    
    def run_complete_demo(self) -> None:
        """运行完整的浮点数非结合性演示"""
        print("开始浮点数非结合性完整演示...\n")
        
        # 1. 基本非结合性演示
        self.demonstrate_non_associativity()
        
        # 2. 求和顺序依赖性演示
        self.demonstrate_sum_order_dependency()
        
        # 3. 矩阵乘法确定性演示
        self.demonstrate_matrix_multiplication_determinism()
        
        # 4. 可视化结果
        self.visualize_sum_distribution('experiments/plots/floating_point_demo.png')
        
        print("\n浮点数非结合性演示完成！")

def create_floating_point_examples() -> List[Tuple[str, float, float]]:
    """创建更多浮点数非结合性示例"""
    examples = []
    
    # 示例1: 不同数量级
    a, b, c = 0.1, 1e20, -1e20
    result1 = (a + b) + c
    result2 = a + (b + c)
    examples.append(("不同数量级", result1, result2))
    
    # 示例2: 精度损失
    a, b, c = 1e-10, 1e-5, 1e-2
    result1 = (a + b) + c
    result2 = a + (b + c)
    examples.append(("精度损失", result1, result2))
    
    # 示例3: 大数相加
    a, b, c = 1e15, 1e-10, -1e15
    result1 = (a + b) + c
    result2 = a + (b + c)
    examples.append(("大数相加", result1, result2))
    
    return examples

if __name__ == "__main__":
    demo = FloatingPointDemo()
    demo.run_complete_demo()
