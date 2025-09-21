#!/usr/bin/env python3
"""
LeNet确定性演示
测试简单神经网络在温度0（贪婪采样）情况下的输出一致性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Tuple
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from device_manager import get_device
from font_config import setup_chinese_fonts, force_chinese_fonts

# 设置中文字体
setup_chinese_fonts()
force_chinese_fonts()

class SimpleLeNet(nn.Module):
    """简化的LeNet网络，用于分类任务"""
    
    def __init__(self, input_size: int = 784, num_classes: int = 10):
        super(SimpleLeNet, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)            # 28x28 -> 24x24
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)       # 24x24 -> 12x12
        
        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 确保输入是4D张量 (batch_size, channels, height, width)
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)  # 假设输入是28x28的MNIST风格图像
        
        # 卷积层 + 激活函数 + 池化
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 5x5
        
        # 展平 - 修正尺寸计算
        x = x.view(-1, 16 * 5 * 5)  # 16个通道，5x5的特征图
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class LeNetDeterminismDemo:
    """LeNet确定性演示类"""
    
    def __init__(self, device: str = 'auto'):
        self.device = get_device(device)
        print(f"使用设备: {self.device}")
        
        # 创建模型
        self.model = SimpleLeNet().to(self.device)
        self.model.eval()  # 设置为评估模式
        
        # 创建一些测试数据
        self.test_inputs = self._create_test_data()
        
    def _create_test_data(self) -> List[torch.Tensor]:
        """创建测试数据"""
        test_inputs = []
        
        # 创建几个不同的测试样本
        for i in range(5):
            # 创建随机但固定的输入
            torch.manual_seed(42 + i)
            input_data = torch.randn(1, 784, device=self.device)  # 28x28=784
            test_inputs.append(input_data)
            
        return test_inputs
    
    def set_all_seeds(self, seed: int = 42):
        """设置所有随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == 'mps':
            torch.mps.manual_seed(seed)
        elif self.device.type == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        print(f"所有随机种子已设置为: {seed}")
    
    def reset_all_seeds(self):
        """重置所有随机种子为随机状态"""
        random.seed()
        np.random.seed()
        torch.manual_seed(torch.initial_seed())
        if self.device.type == 'mps':
            torch.mps.manual_seed(random.randint(0, 2**32-1))
        elif self.device.type == 'cuda':
            torch.cuda.manual_seed(torch.cuda.initial_seed())
        print("所有随机种子已重置为随机状态")
    
    def sample_with_temperature(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """带温度的采样"""
        if temperature == 0:
            # 贪婪采样：选择概率最高的类别
            return torch.argmax(logits, dim=-1)
        else:
            # 温度采样
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)
    
    def test_determinism_with_temperature(self, input_data: torch.Tensor, 
                                        temperature: float = 0.0, 
                                        num_trials: int = 10) -> List[int]:
        """测试特定温度下的确定性"""
        print(f"\n=== 温度={temperature}时的输出确定性测试 ===")
        
        results = []
        
        for trial in range(num_trials):
            if temperature == 0:
                # 温度=0时，固定种子确保确定性
                self.set_all_seeds(42)
            else:
                # 温度>0时，使用不同的种子来展示随机性
                self.set_all_seeds(42 + trial)
            
            # 前向传播
            with torch.no_grad():
                logits = self.model(input_data)
                predicted_class = self.sample_with_temperature(logits, temperature=temperature)
                results.append(predicted_class.item())
            
            print(f"  第{trial+1}次预测: 类别 = {results[-1]}")
        
        # 分析结果
        unique_classes = set(results)
        print(f"\n=== 分析结果 ===")
        print(f"唯一类别数量: {len(unique_classes)}")
        print(f"唯一类别: {sorted(unique_classes)}")
        
        if temperature == 0:
            is_deterministic = len(unique_classes) == 1
            print(f"温度=0时是否确定性: {is_deterministic}")
        else:
            print(f"温度={temperature}时的随机性: {len(unique_classes)}/{num_trials}")
        
        return results
    
    def test_inference_invariance(self, input_data: torch.Tensor, num_trials: int = 5) -> bool:
        """测试推理入口不变性"""
        print(f"\n=== 推理入口不变性测试 ===")
        
        # 固定种子
        self.set_all_seeds(42)
        
        # 获取参考输出
        with torch.no_grad():
            reference_logits = self.model(input_data)
            reference_output = torch.argmax(reference_logits, dim=-1).item()
        
        print(f"参考输出: 类别 = {reference_output}")
        
        # 多次推理测试
        all_consistent = True
        for trial in range(num_trials):
            # 每次推理前都设置相同的种子
            self.set_all_seeds(42)
            
            with torch.no_grad():
                logits = self.model(input_data)
                output = torch.argmax(logits, dim=-1).item()
            
            is_consistent = (output == reference_output)
            all_consistent = all_consistent and is_consistent
            
            print(f"  第{trial+1}次推理: 类别 = {output}, 一致性 = {is_consistent}")
        
        print(f"\n推理入口不变性: {'通过' if all_consistent else '失败'}")
        return all_consistent
    
    def visualize_temperature_effects(self, input_data: torch.Tensor, 
                                    temperatures: List[float] = [0.0, 0.5, 1.0, 2.0]):
        """可视化温度对输出的影响"""
        print(f"\n=== 可视化温度对输出的影响 ===")
        
        # 获取logits
        with torch.no_grad():
            logits = self.model(input_data)
            logits_np = logits.cpu().numpy()[0]
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, temp in enumerate(temperatures):
            # 计算概率分布
            if temp == 0:
                # 温度=0时，使用one-hot编码
                probs = np.zeros_like(logits_np)
                probs[np.argmax(logits_np)] = 1.0
            else:
                # 温度>0时，使用softmax
                probs = F.softmax(torch.tensor(logits_np) / temp, dim=-1).numpy()
            
            # 获取top-5概率
            top_indices = np.argsort(probs)[-5:][::-1]
            top_probs = probs[top_indices]
            
            # 绘制柱状图
            axes[i].bar(range(len(top_probs)), top_probs, alpha=0.7, 
                       color=plt.cm.viridis(np.linspace(0, 1, len(top_probs))))
            axes[i].set_title(f'温度 = {temp}', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('类别排名', fontsize=12)
            axes[i].set_ylabel('概率', fontsize=12)
            axes[i].set_xticks(range(len(top_probs)))
            axes[i].set_xticklabels([f'类别{idx}' for idx in top_indices])
            axes[i].grid(True, alpha=0.3)
            
            # 添加数值标签
            for j, prob in enumerate(top_probs):
                axes[i].text(j, prob + 0.01, f'{prob:.3f}', 
                           ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.suptitle('温度对LeNet输出概率分布的影响', fontsize=16, fontweight='bold', y=1.02)
        
        # 保存图片
        os.makedirs('experiments/plots', exist_ok=True)
        plt.savefig('experiments/plots/lenet_temperature_effects.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("温度效果可视化已保存到: experiments/plots/lenet_temperature_effects.png")
    
    def comprehensive_determinism_test(self):
        """综合确定性测试"""
        print("=" * 60)
        print("LeNet确定性综合测试")
        print("=" * 60)
        
        # 测试第一个输入样本
        test_input = self.test_inputs[0]
        
        # 1. 推理入口不变性测试
        inference_consistent = self.test_inference_invariance(test_input)
        
        # 2. 温度=0确定性测试
        temp0_results = self.test_determinism_with_temperature(test_input, temperature=0.0)
        
        # 3. 温度>0随机性测试
        temp1_results = self.test_determinism_with_temperature(test_input, temperature=1.0)
        
        # 4. 可视化温度效果
        self.visualize_temperature_effects(test_input)
        
        # 5. 总结
        print("\n" + "=" * 60)
        print("测试总结")
        print("=" * 60)
        print(f"推理入口不变性: {'✅ 通过' if inference_consistent else '❌ 失败'}")
        print(f"温度=0确定性: {'✅ 通过' if len(set(temp0_results)) == 1 else '❌ 失败'}")
        print(f"温度=1.0随机性: {'✅ 正常' if len(set(temp1_results)) > 1 else '❌ 异常'}")
        
        return {
            'inference_consistent': inference_consistent,
            'temp0_deterministic': len(set(temp0_results)) == 1,
            'temp1_random': len(set(temp1_results)) > 1
        }

def main():
    """主函数"""
    print("LeNet确定性演示")
    print("=" * 40)
    
    # 创建演示实例
    demo = LeNetDeterminismDemo()
    
    # 运行综合测试
    results = demo.comprehensive_determinism_test()
    
    print("\n🎉 LeNet确定性测试完成！")
    print("关键发现:")
    print("- 在温度=0（贪婪采样）时，相同输入应该产生相同输出")
    print("- 在温度>0时，相同输入可能产生不同输出（这是正常的随机性）")
    print("- 推理入口不变性确保模型行为的一致性")

if __name__ == "__main__":
    main()
