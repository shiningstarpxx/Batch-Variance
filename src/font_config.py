"""
字体配置工具

这个模块用于配置matplotlib的中文字体支持，解决中文显示乱码问题。
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import os

def setup_chinese_fonts():
    """设置中文字体支持"""
    
    # 获取系统信息
    system = platform.system()
    
    # 定义不同系统的中文字体优先级
    if system == "Darwin":  # macOS
        font_candidates = [
            'Arial Unicode MS',
            'STHeiti',
            'STHeiti Light',
            'PingFang SC',
            'Hiragino Sans GB',
            'SimHei',
            'Microsoft YaHei'
        ]
    elif system == "Windows":
        font_candidates = [
            'Microsoft YaHei',
            'SimHei',
            'SimSun',
            'KaiTi',
            'Arial Unicode MS'
        ]
    else:  # Linux
        font_candidates = [
            'WenQuanYi Micro Hei',
            'WenQuanYi Zen Hei',
            'Noto Sans CJK SC',
            'Source Han Sans SC',
            'SimHei',
            'Arial Unicode MS'
        ]
    
    # 查找可用的中文字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    selected_font = None
    
    for font in font_candidates:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font:
        print(f"使用中文字体: {selected_font}")
        # 设置字体优先级，确保中文字体在最前面
        plt.rcParams['font.sans-serif'] = [selected_font] + [f for f in font_candidates if f != selected_font] + ['DejaVu Sans', 'Arial', 'sans-serif']
    else:
        print("警告: 未找到合适的中文字体，将使用默认字体")
        # 尝试使用系统默认字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    
    # 设置负号显示
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设置字体大小
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    
    return selected_font

def force_chinese_fonts():
    """强制设置中文字体，确保不被其他设置覆盖"""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        chinese_fonts = ['Arial Unicode MS', 'STHeiti', 'PingFang SC', 'Hiragino Sans GB']
    elif system == "Windows":
        chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun']
    else:  # Linux
        chinese_fonts = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Source Han Sans SC']
    
    # 强制设置字体
    plt.rcParams['font.sans-serif'] = chinese_fonts + ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    return chinese_fonts[0] if chinese_fonts else 'DejaVu Sans'

def test_chinese_display():
    """测试中文显示效果"""
    import numpy as np
    
    # 创建测试图表
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    ax.plot(x, y, label='正弦函数')
    ax.set_title('中文字体测试图表')
    ax.set_xlabel('横坐标 (x)')
    ax.set_ylabel('纵坐标 (y)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加中文文本
    ax.text(5, 0.5, '这是中文测试文本', fontsize=14, ha='center')
    
    plt.tight_layout()
    plt.savefig('experiments/plots/chinese_font_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("中文字体测试图表已保存到: experiments/plots/chinese_font_test.png")

def get_font_info():
    """获取字体信息"""
    print("=== 字体信息 ===")
    print(f"当前字体设置: {plt.rcParams['font.sans-serif']}")
    print(f"负号显示: {plt.rcParams['axes.unicode_minus']}")
    
    # 列出所有可用字体
    fonts = [f.name for f in fm.fontManager.ttflist]
    chinese_fonts = [f for f in fonts if any(keyword in f for keyword in 
                   ['Chinese', 'SimHei', 'Arial Unicode', 'STHeiti', 'PingFang', 'Hiragino', 'WenQuanYi', 'Noto', 'Source Han'])]
    
    print(f"\n可用的中文字体: {chinese_fonts}")
    
    return chinese_fonts

if __name__ == "__main__":
    # 设置中文字体
    selected_font = setup_chinese_fonts()
    
    # 获取字体信息
    get_font_info()
    
    # 测试中文显示
    test_chinese_display()
