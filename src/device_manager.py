"""
设备管理模块

这个模块用于管理计算设备，支持CPU、CUDA和MPS（Metal Performance Shaders）。
特别针对Mac的Apple Silicon芯片进行了优化。
"""

import torch
import platform
import psutil
from typing import Optional, Dict, Any
import time

class DeviceManager:
    """设备管理器类"""
    
    def __init__(self):
        self.system = platform.system()
        self.device_info = self._get_device_info()
        self.optimal_device = self._select_optimal_device()
        
    def _get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        info = {
            'system': self.system,
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'cuda_available': torch.cuda.is_available(),
            'mps_available': False,
            'cuda_device_count': 0,
            'cuda_device_name': None
        }
        
        # 检查MPS支持
        if hasattr(torch.backends, 'mps'):
            info['mps_available'] = torch.backends.mps.is_available()
        
        # 检查CUDA设备
        if info['cuda_available']:
            info['cuda_device_count'] = torch.cuda.device_count()
            info['cuda_device_name'] = torch.cuda.get_device_name(0)
        
        return info
    
    def _select_optimal_device(self) -> str:
        """选择最优计算设备"""
        if self.device_info['cuda_available']:
            return 'cuda'
        elif self.device_info['mps_available']:
            return 'mps'
        else:
            return 'cpu'
    
    def get_device(self, device: Optional[str] = None) -> torch.device:
        """获取计算设备"""
        if device is None:
            device = self.optimal_device
        
        if device == 'cuda' and not self.device_info['cuda_available']:
            print("警告: CUDA不可用，回退到CPU")
            device = 'cpu'
        elif device == 'mps' and not self.device_info['mps_available']:
            print("警告: MPS不可用，回退到CPU")
            device = 'cpu'
        
        return torch.device(device)
    
    def benchmark_device(self, device: str, size: int = 1000) -> Dict[str, float]:
        """基准测试设备性能"""
        device_obj = self.get_device(device)
        
        # 创建测试数据
        a = torch.randn(size, size, device=device_obj)
        b = torch.randn(size, size, device=device_obj)
        
        # 预热
        for _ in range(5):
            _ = torch.matmul(a, b)
        
        # 同步设备
        if device_obj.type == 'cuda':
            torch.cuda.synchronize()
        elif device_obj.type == 'mps':
            torch.mps.synchronize()
        
        # 基准测试
        start_time = time.time()
        for _ in range(10):
            result = torch.matmul(a, b)
        end_time = time.time()
        
        # 同步设备
        if device_obj.type == 'cuda':
            torch.cuda.synchronize()
        elif device_obj.type == 'mps':
            torch.mps.synchronize()
        
        avg_time = (end_time - start_time) / 10
        
        return {
            'device': device,
            'avg_time_ms': avg_time * 1000,
            'matrix_size': size,
            'operations_per_second': (size ** 3) / avg_time
        }
    
    def benchmark_all_devices(self, size: int = 1000) -> Dict[str, Dict[str, float]]:
        """基准测试所有可用设备"""
        results = {}
        
        # 测试CPU
        try:
            results['cpu'] = self.benchmark_device('cpu', size)
        except Exception as e:
            print(f"CPU基准测试失败: {e}")
        
        # 测试CUDA
        if self.device_info['cuda_available']:
            try:
                results['cuda'] = self.benchmark_device('cuda', size)
            except Exception as e:
                print(f"CUDA基准测试失败: {e}")
        
        # 测试MPS
        if self.device_info['mps_available']:
            try:
                results['mps'] = self.benchmark_device('mps', size)
            except Exception as e:
                print(f"MPS基准测试失败: {e}")
        
        return results
    
    def get_memory_info(self, device: str) -> Dict[str, Any]:
        """获取设备内存信息"""
        device_obj = self.get_device(device)
        info = {'device': device}
        
        if device_obj.type == 'cuda':
            info['total_memory'] = torch.cuda.get_device_properties(0).total_memory
            info['allocated_memory'] = torch.cuda.memory_allocated(0)
            info['cached_memory'] = torch.cuda.memory_reserved(0)
        elif device_obj.type == 'mps':
            # MPS内存信息获取
            info['total_memory'] = self.device_info['memory_total']
            info['allocated_memory'] = 0  # MPS不提供详细内存信息
            info['cached_memory'] = 0
        else:
            info['total_memory'] = self.device_info['memory_total']
            info['allocated_memory'] = self.device_info['memory_total'] - self.device_info['memory_available']
            info['cached_memory'] = 0
        
        return info
    
    def print_device_info(self):
        """打印设备信息"""
        print("=== 设备信息 ===")
        print(f"系统: {self.device_info['system']}")
        print(f"CPU核心数: {self.device_info['cpu_count']}")
        print(f"总内存: {self.device_info['memory_total'] / (1024**3):.2f} GB")
        print(f"可用内存: {self.device_info['memory_available'] / (1024**3):.2f} GB")
        print(f"CUDA可用: {self.device_info['cuda_available']}")
        print(f"MPS可用: {self.device_info['mps_available']}")
        
        if self.device_info['cuda_available']:
            print(f"CUDA设备数: {self.device_info['cuda_device_count']}")
            print(f"CUDA设备名: {self.device_info['cuda_device_name']}")
        
        print(f"推荐设备: {self.optimal_device}")
        print()

# 全局设备管理器实例
device_manager = DeviceManager()

def get_device(device: Optional[str] = None) -> torch.device:
    """获取计算设备的便捷函数"""
    return device_manager.get_device(device)

def benchmark_devices(size: int = 1000) -> Dict[str, Dict[str, float]]:
    """基准测试所有设备的便捷函数"""
    return device_manager.benchmark_all_devices(size)

def print_device_info():
    """打印设备信息的便捷函数"""
    device_manager.print_device_info()
