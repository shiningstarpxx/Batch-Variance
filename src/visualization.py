"""
可视化工具模块

这个模块提供了用于可视化LLM推理非确定性研究结果的各种图表和工具。
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# 导入字体配置
try:
    from .font_config import setup_chinese_fonts, force_chinese_fonts
except ImportError:
    from font_config import setup_chinese_fonts, force_chinese_fonts

# 设置中文字体
setup_chinese_fonts()

# 设置其他样式（在字体设置之后）
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")

# 强制设置中文字体（seaborn可能会覆盖字体设置）
force_chinese_fonts()

class VisualizationUtils:
    """可视化工具类"""
    
    def __init__(self):
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
        self.chinese_fonts = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'Microsoft YaHei']
    
    def create_comprehensive_dashboard(self, data: Dict[str, Any], 
                                     save_path: str = 'experiments/plots/comprehensive_dashboard.png') -> None:
        """创建综合仪表板"""
        # 确保中文字体设置
        force_chinese_fonts()
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('LLM推理非确定性研究综合仪表板', fontsize=20, fontweight='bold')
        
        # 子图1: 浮点数非结合性示例
        ax1 = axes[0, 0]
        self._plot_floating_point_examples(ax1)
        
        # 子图2: 注意力机制差异分布
        ax2 = axes[0, 1]
        self._plot_attention_differences(ax2, data.get('attention_diffs', []))
        
        # 子图3: 批处理不变性对比
        ax3 = axes[0, 2]
        self._plot_batch_invariance_comparison(ax3, data.get('batch_invariance', {}))
        
        # 子图4: 性能对比
        ax4 = axes[1, 0]
        self._plot_performance_comparison(ax4, data.get('performance', {}))
        
        # 子图5: 数值稳定性分析
        ax5 = axes[1, 1]
        self._plot_numerical_stability(ax5, data.get('stability', {}))
        
        # 子图6: 总结统计
        ax6 = axes[1, 2]
        self._plot_summary_statistics(ax6, data.get('summary', {}))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"综合仪表板已保存到: {save_path}")
    
    def _plot_floating_point_examples(self, ax: plt.Axes) -> None:
        """绘制浮点数非结合性示例"""
        # 创建示例数据
        examples = [
            ("不同数量级", 0.1, 1e20, -1e20),
            ("精度损失", 1e-10, 1e-5, 1e-2),
            ("大数相加", 1e15, 1e-10, -1e15)
        ]
        
        results = []
        labels = []
        
        for label, a, b, c in examples:
            result1 = (a + b) + c
            result2 = a + (b + c)
            diff = abs(result1 - result2)
            results.append(diff)
            labels.append(label)
        
        bars = ax.bar(labels, results, color=[self.colors['primary'], self.colors['secondary'], self.colors['danger']])
        ax.set_title('浮点数非结合性示例', fontsize=14, fontweight='bold')
        ax.set_ylabel('数值差异', fontsize=12)
        ax.set_yscale('log')
        ax.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, result in zip(bars, results):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{result:.2e}', ha='center', va='bottom', fontsize=10)
    
    def _plot_attention_differences(self, ax: plt.Axes, attention_diffs: List[float]) -> None:
        """绘制注意力机制差异分布"""
        if len(attention_diffs) == 0:
            # 创建示例数据
            attention_diffs = np.random.exponential(1e-6, 1000)
        
        ax.hist(attention_diffs, bins=50, alpha=0.7, color=self.colors['info'], 
               edgecolor='black', density=True)
        ax.set_title('注意力机制差异分布', fontsize=14, fontweight='bold')
        ax.set_xlabel('平均绝对差异', fontsize=12)
        ax.set_ylabel('密度', fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_diff = np.mean(attention_diffs)
        ax.axvline(mean_diff, color=self.colors['danger'], linestyle='--', 
                  linewidth=2, label=f'平均值: {mean_diff:.2e}')
        ax.legend()
    
    def _plot_batch_invariance_comparison(self, ax: plt.Axes, batch_data: Dict[str, List[float]]) -> None:
        """绘制批处理不变性对比"""
        if not batch_data:
            # 创建示例数据
            seq_lengths = [64, 128, 256, 512, 1024]
            standard_diffs = [1e-6, 2e-6, 4e-6, 8e-6, 1.6e-5]
            batch_invariant_diffs = [1e-8, 1e-8, 1e-8, 1e-8, 1e-8]
        else:
            seq_lengths = batch_data.get('seq_lengths', [64, 128, 256, 512, 1024])
            standard_diffs = batch_data.get('standard_diffs', [1e-6, 2e-6, 4e-6, 8e-6, 1.6e-5])
            batch_invariant_diffs = batch_data.get('batch_invariant_diffs', [1e-8, 1e-8, 1e-8, 1e-8, 1e-8])
        
        ax.plot(seq_lengths, standard_diffs, 'o-', label='标准实现', 
               linewidth=2, markersize=8, color=self.colors['danger'])
        ax.plot(seq_lengths, batch_invariant_diffs, 's-', label='批处理不变性', 
               linewidth=2, markersize=8, color=self.colors['success'])
        
        ax.set_title('批处理不变性对比', fontsize=14, fontweight='bold')
        ax.set_xlabel('序列长度', fontsize=12)
        ax.set_ylabel('平均绝对差异', fontsize=12)
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_performance_comparison(self, ax: plt.Axes, performance_data: Dict[str, List[float]]) -> None:
        """绘制性能对比"""
        if not performance_data:
            # 创建示例数据
            methods = ['标准注意力', 'Split-KV', '批处理不变性']
            times = [10, 12, 15]  # 毫秒
        else:
            methods = performance_data.get('methods', ['标准注意力', 'Split-KV', '批处理不变性'])
            times = performance_data.get('times', [10, 12, 15])
        
        bars = ax.bar(methods, times, color=[self.colors['primary'], self.colors['warning'], self.colors['success']])
        ax.set_title('性能对比', fontsize=14, fontweight='bold')
        ax.set_ylabel('执行时间 (ms)', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time}ms', ha='center', va='bottom', fontsize=10)
    
    def _plot_numerical_stability(self, ax: plt.Axes, stability_data: Dict[str, Any]) -> None:
        """绘制数值稳定性分析"""
        if not stability_data:
            # 创建示例数据
            iterations = range(100)
            standard_values = np.cumsum(np.random.normal(0, 1e-6, 100))
            batch_invariant_values = np.cumsum(np.random.normal(0, 1e-8, 100))
        else:
            iterations = stability_data.get('iterations', range(100))
            standard_values = stability_data.get('standard_values', np.cumsum(np.random.normal(0, 1e-6, 100)))
            batch_invariant_values = stability_data.get('batch_invariant_values', np.cumsum(np.random.normal(0, 1e-8, 100)))
        
        ax.plot(iterations, standard_values, label='标准实现', 
               linewidth=2, color=self.colors['danger'])
        ax.plot(iterations, batch_invariant_values, label='批处理不变性', 
               linewidth=2, color=self.colors['success'])
        
        ax.set_title('数值稳定性分析', fontsize=14, fontweight='bold')
        ax.set_xlabel('迭代次数', fontsize=12)
        ax.set_ylabel('累积误差', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_summary_statistics(self, ax: plt.Axes, summary_data: Dict[str, float]) -> None:
        """绘制总结统计"""
        if not summary_data:
            # 创建示例数据
            metrics = ['确定性\n提升', '性能\n开销', '数值\n稳定性', '批处理\n一致性']
            values = [95, 20, 90, 99]  # 百分比
        else:
            metrics = summary_data.get('metrics', ['确定性\n提升', '性能\n开销', '数值\n稳定性', '批处理\n一致性'])
            values = summary_data.get('values', [95, 20, 90, 99])
        
        # 创建雷达图样式
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values_plot = values + [values[0]]  # 闭合图形
        angles += angles[:1]
        
        ax.plot(angles, values_plot, 'o-', linewidth=2, color=self.colors['primary'])
        ax.fill(angles, values_plot, alpha=0.25, color=self.colors['primary'])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 100)
        ax.set_title('总结统计', fontsize=14, fontweight='bold')
        ax.grid(True)
    
    def create_interactive_plot(self, data: Dict[str, Any], 
                              save_path: str = 'experiments/plots/interactive_plot.html') -> None:
        """创建交互式图表"""
        # 确保中文字体设置
        force_chinese_fonts()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('浮点数非结合性', '注意力机制差异', '批处理不变性', '性能对比'),
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 子图1: 浮点数非结合性
        examples = [0.1, 1e-10, 1e15]
        results = [abs((ex + 1e20) + (-1e20) - (ex + (1e20 + (-1e20)))) for ex in examples]
        fig.add_trace(
            go.Bar(x=['不同数量级', '精度损失', '大数相加'], y=results, name='数值差异'),
            row=1, col=1
        )
        
        # 子图2: 注意力机制差异分布
        attention_diffs = np.random.exponential(1e-6, 1000)
        fig.add_trace(
            go.Histogram(x=attention_diffs, name='差异分布', nbinsx=50),
            row=1, col=2
        )
        
        # 子图3: 批处理不变性
        seq_lengths = [64, 128, 256, 512, 1024]
        standard_diffs = [1e-6, 2e-6, 4e-6, 8e-6, 1.6e-5]
        batch_invariant_diffs = [1e-8, 1e-8, 1e-8, 1e-8, 1e-8]
        
        fig.add_trace(
            go.Scatter(x=seq_lengths, y=standard_diffs, mode='lines+markers', 
                      name='标准实现', line=dict(color='red')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=seq_lengths, y=batch_invariant_diffs, mode='lines+markers', 
                      name='批处理不变性', line=dict(color='green')),
            row=2, col=1
        )
        
        # 子图4: 性能对比
        methods = ['标准注意力', 'Split-KV', '批处理不变性']
        times = [10, 12, 15]
        fig.add_trace(
            go.Bar(x=methods, y=times, name='执行时间', marker_color=['blue', 'orange', 'green']),
            row=2, col=2
        )
        
        # 更新布局
        fig.update_layout(
            title_text="LLM推理非确定性研究交互式仪表板",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        # 保存交互式图表
        fig.write_html(save_path)
        print(f"交互式图表已保存到: {save_path}")
    
    def create_heatmap(self, data: np.ndarray, title: str = "热力图", 
                      save_path: str = None) -> None:
        """创建热力图"""
        # 确保中文字体设置
        force_chinese_fonts()
        
        plt.figure(figsize=(10, 8))
        
        if data is None:
            # 创建示例数据
            data = np.random.rand(10, 10)
        
        sns.heatmap(data, annot=True, fmt='.2e', cmap='viridis', 
                   cbar_kws={'label': '数值'})
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"热力图已保存到: {save_path}")
        
        plt.show()
    
    def create_correlation_matrix(self, data: pd.DataFrame, 
                                save_path: str = None) -> None:
        """创建相关性矩阵"""
        # 确保中文字体设置
        force_chinese_fonts()
        
        plt.figure(figsize=(10, 8))
        
        if data is None or data.empty:
            # 创建示例数据
            np.random.seed(42)
            data = pd.DataFrame({
                '浮点数差异': np.random.exponential(1e-6, 100),
                '注意力差异': np.random.exponential(1e-5, 100),
                '批处理差异': np.random.exponential(1e-7, 100),
                '性能开销': np.random.normal(1.2, 0.1, 100)
            })
        
        correlation_matrix = data.corr()
        
        sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, square=True, cbar_kws={'label': '相关系数'})
        plt.title('变量相关性矩阵', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"相关性矩阵已保存到: {save_path}")
        
        plt.show()
    
    def create_3d_surface(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                         title: str = "3D表面图", save_path: str = None) -> None:
        """创建3D表面图"""
        # 确保中文字体设置
        force_chinese_fonts()
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if x is None or y is None or z is None:
            # 创建示例数据
            x = np.linspace(-5, 5, 50)
            y = np.linspace(-5, 5, 50)
            X, Y = np.meshgrid(x, y)
            Z = np.sin(np.sqrt(X**2 + Y**2))
        else:
            X, Y = np.meshgrid(x, y)
            Z = z
        
        surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax.set_xlabel('X轴', fontsize=12)
        ax.set_ylabel('Y轴', fontsize=12)
        ax.set_zlabel('Z轴', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # 添加颜色条
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D表面图已保存到: {save_path}")
        
        plt.show()
    
    def create_animation_frames(self, data_sequence: List[np.ndarray], 
                              save_dir: str = 'experiments/plots/animation_frames/') -> None:
        """创建动画帧"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for i, data in enumerate(data_sequence):
            plt.figure(figsize=(8, 6))
            plt.plot(data, linewidth=2, color=self.colors['primary'])
            plt.title(f'动画帧 {i+1}', fontsize=14, fontweight='bold')
            plt.xlabel('索引', fontsize=12)
            plt.ylabel('数值', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            frame_path = os.path.join(save_dir, f'frame_{i:03d}.png')
            plt.savefig(frame_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"动画帧已保存到: {save_dir}")

def create_sample_data() -> Dict[str, Any]:
    """创建示例数据用于演示"""
    return {
        'attention_diffs': np.random.exponential(1e-6, 1000),
        'batch_invariance': {
            'seq_lengths': [64, 128, 256, 512, 1024],
            'standard_diffs': [1e-6, 2e-6, 4e-6, 8e-6, 1.6e-5],
            'batch_invariant_diffs': [1e-8, 1e-8, 1e-8, 1e-8, 1e-8]
        },
        'performance': {
            'methods': ['标准注意力', 'Split-KV', '批处理不变性'],
            'times': [10, 12, 15]
        },
        'stability': {
            'iterations': range(100),
            'standard_values': np.cumsum(np.random.normal(0, 1e-6, 100)),
            'batch_invariant_values': np.cumsum(np.random.normal(0, 1e-8, 100))
        },
        'summary': {
            'metrics': ['确定性\n提升', '性能\n开销', '数值\n稳定性', '批处理\n一致性'],
            'values': [95, 20, 90, 99]
        }
    }

if __name__ == "__main__":
    # 创建可视化工具实例
    viz = VisualizationUtils()
    
    # 创建示例数据
    sample_data = create_sample_data()
    
    # 创建综合仪表板
    viz.create_comprehensive_dashboard(sample_data)
    
    # 创建交互式图表
    viz.create_interactive_plot(sample_data)
    
    # 创建其他图表
    viz.create_heatmap(None, "示例热力图", "experiments/plots/heatmap.png")
    viz.create_correlation_matrix(None, "experiments/plots/correlation_matrix.png")
    viz.create_3d_surface(None, None, None, "示例3D表面图", "experiments/plots/3d_surface.png")
    
    print("所有可视化图表创建完成！")
