# LLM推理非确定性复现项目

本项目复现了Thinking Machines关于LLM推理非确定性的研究，特别针对Mac环境进行了优化。

## 项目背景

大型语言模型(LLM)在推理过程中存在非确定性问题，即使设置temperature=0（贪婪采样）也无法保证完全确定性的输出。本项目深入分析了这一现象的根本原因，并提供了解决方案。

## 主要发现

1. **浮点数非结合性**：`(a + b) + c ≠ a + (b + c)`
2. **批处理非不变性**：不同批处理大小导致不同的数值结果
3. **注意力机制的非确定性**：Split-KV策略导致的数值差异

## 项目结构

```
batch-variance/
├── requirements.txt          # 项目依赖
├── README.md                # 项目说明
├── notebooks/               # Jupyter notebooks
│   ├── 01_floating_point_demo.ipynb
│   ├── 02_attention_nondeterminism.ipynb
│   ├── 03_batch_invariant_solution.ipynb
│   └── 04_experiments.ipynb
├── src/                     # 源代码
│   ├── __init__.py
│   ├── floating_point.py    # 浮点数非结合性演示
│   ├── attention.py         # 注意力机制实现
│   ├── batch_invariant.py   # 批处理不变性解决方案
│   └── visualization.py     # 可视化工具
└── experiments/             # 实验结果
    ├── results/
    └── plots/
```

## 安装和使用

### 方法1：使用安装脚本（推荐）

1. 运行安装脚本：
```bash
chmod +x install.sh
./install.sh
```

2. 激活虚拟环境（如果创建了）：
```bash
source venv/bin/activate
```

### 方法2：手动安装

1. 创建虚拟环境：
```bash
python3 -m venv venv
source venv/bin/activate
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

### 运行演示

1. 运行所有演示：
```bash
python run_demo.py --demo all
```

2. 运行特定演示：
```bash
python run_demo.py --demo floating_point    # 浮点数非结合性演示
python run_demo.py --demo attention         # 注意力机制非确定性演示
python run_demo.py --demo batch_invariant   # 批处理不变性演示
python run_demo.py --demo visualization     # 可视化演示
```

3. 运行Jupyter notebooks：
```bash
jupyter notebook notebooks/
```

## 主要功能

- 浮点数非结合性演示
- 注意力机制非确定性分析
- 批处理不变性解决方案
- 性能对比实验
- 中文可视化图表

## 实验结果

运行演示后，结果将保存在 `experiments/plots/` 目录中：

- `floating_point_demo.png`: 浮点数非结合性演示图表
- `attention_differences.png`: 注意力机制差异分布
- `attention_performance.png`: 注意力机制性能对比
- `batch_invariance.png`: 批处理不变性对比
- `batch_invariant_performance.png`: 批处理不变性性能对比
- `comprehensive_dashboard.png`: 综合仪表板
- `interactive_dashboard.html`: 交互式仪表板（可在浏览器中打开）

## 主要发现

1. **浮点数非结合性**：验证了浮点数运算的非结合性特性，这是导致LLM推理非确定性的根本原因。

2. **注意力机制非确定性**：虽然我们的简化实现没有显示明显的非确定性，但在实际的大规模并行计算中，Split-KV策略会导致数值差异。

3. **批处理不变性解决方案**：通过固定分割大小和计算顺序，可以显著减少数值差异，但会带来一定的性能开销。

4. **性能权衡**：批处理不变性解决方案的性能开销在可接受范围内，特别是在需要确定性的场景中。

5. **中文字体支持**：项目已完全支持中文显示，所有图表和界面都使用正确的中文字体渲染。

## 字体配置

项目包含完整的中文字体配置系统：

- **自动字体检测**：自动检测系统中可用的中文字体
- **跨平台支持**：支持macOS、Windows和Linux系统
- **强制字体设置**：确保seaborn等库不会覆盖中文字体设置
- **字体测试工具**：提供字体显示验证功能

### 测试中文字体

```bash
# 运行字体测试
python test_chinese_fonts.py
```

## 参考文献

He, Horace and Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference", Thinking Machines Lab: Connectionism, Sep 2025.
