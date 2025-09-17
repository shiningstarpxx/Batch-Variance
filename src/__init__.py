"""
LLM推理非确定性复现项目

这个包包含了复现Thinking Machines关于LLM推理非确定性研究的所有核心功能。
"""

__version__ = "1.0.0"
__author__ = "复现项目"

# 首先设置中文字体
from .font_config import setup_chinese_fonts
setup_chinese_fonts()

from .device_manager import *
from .floating_point import *
from .attention import *
from .batch_invariant import *
from .visualization import *
from .advanced_analysis import *
