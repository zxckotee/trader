"""
Device management utilities for CPU/GPU switching.
"""

import torch
import os
from typing import Tuple, Optional


class DeviceManager:
    """
    Manages device selection and configuration for CPU/GPU usage.
    """
    
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        
    def get_optimal_device(self, 
                          force_cpu: bool = False,
                          device_preference: str = 'auto') -> torch.device:
        """
        Get optimal device based on availability and preferences.
        
        Args:
            force_cpu: Force CPU usage even if GPU is available
            device_preference: 'auto', 'cpu', 'cuda', 'mps'
            
        Returns:
            torch.device object
        """
        if force_cpu or device_preference == 'cpu':
            return torch.device('cpu')
        
        if device_preference == 'cuda' and self.cuda_available:
            return torch.device('cuda')
        
        if device_preference == 'mps' and self.mps_available:
            return torch.device('mps')
        
        # Auto selection
        if device_preference == 'auto':
            if self.cuda_available:
                return torch.device('cuda')
            elif self.mps_available:
                return torch.device('mps')
            else:
                return torch.device('cpu')
        
        # Fallback to CPU
        return torch.device('cpu')
    
    def get_optimal_workers(self, 
                           device: torch.device,
                           num_workers: str = 'auto') -> int:
        """
        Get optimal number of workers based on device and system.
        
        Args:
            device: Target device
            num_workers: 'auto' or specific number
            
        Returns:
            Number of workers
        """
        if isinstance(num_workers, int):
            return num_workers
        
        if num_workers == 'auto':
            if device.type == 'cpu':
                # For CPU, use fewer workers to avoid overhead
                return min(4, os.cpu_count() or 2)
            else:
                # For GPU, can use more workers
                return min(8, os.cpu_count() or 4)
        
        return 2  # Default fallback
    
    def get_memory_efficient_batch_size(self, 
                                      device: torch.device,
                                      base_batch_size: int = 32) -> int:
        """
        Get memory-efficient batch size based on device.
        
        Args:
            device: Target device
            base_batch_size: Base batch size
            
        Returns:
            Adjusted batch size
        """
        if device.type == 'cpu':
            # CPU can handle larger batches but slower
            return base_batch_size
        elif device.type == 'cuda':
            # Check GPU memory
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                # Adjust batch size based on GPU memory (rough estimation)
                if gpu_memory < 4 * 1024**3:  # < 4GB
                    return max(8, base_batch_size // 4)
                elif gpu_memory < 8 * 1024**3:  # < 8GB
                    return max(16, base_batch_size // 2)
                else:  # >= 8GB
                    return base_batch_size
        
        return base_batch_size
    
    def configure_for_device(self, device: torch.device) -> dict:
        """
        Get device-specific configuration.
        
        Args:
            device: Target device
            
        Returns:
            Configuration dictionary
        """
        config = {
            'device': device,
            'pin_memory': device.type in ['cuda', 'mps'],
            'non_blocking': device.type in ['cuda', 'mps']
        }
        
        # Device-specific optimizations
        if device.type == 'cpu':
            # CPU optimizations
            torch.set_num_threads(os.cpu_count() or 4)
            config['compile_model'] = False
        elif device.type == 'cuda':
            # CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            config['compile_model'] = True
        elif device.type == 'mps':
            # MPS optimizations (Apple Silicon)
            config['compile_model'] = False
        
        return config
    
    def print_device_info(self, device: torch.device) -> None:
        """Print detailed device information."""
        print(f"🖥️  Device Information:")
        print(f"   Selected device: {device}")
        print(f"   CUDA available: {self.cuda_available}")
        
        if self.cuda_available and device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   GPU: {gpu_name}")
            print(f"   GPU Memory: {gpu_memory:.1f} GB")
        
        if self.mps_available:
            print(f"   MPS (Apple Silicon) available: {self.mps_available}")
        
        print(f"   CPU cores: {os.cpu_count()}")
        print(f"   PyTorch version: {torch.__version__}")


def setup_device_from_config(config: dict) -> Tuple[torch.device, dict]:
    """
    Setup device configuration from config dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (device, device_config)
    """
    manager = DeviceManager()
    
    # Get device preferences from config
    force_cpu = config.get('training', {}).get('force_cpu', False)
    device_preference = config.get('training', {}).get('device', 'auto')
    
    # Get optimal device
    device = manager.get_optimal_device(force_cpu, device_preference)
    
    # Get device configuration
    device_config = manager.configure_for_device(device)
    
    # Update batch size and workers
    base_batch_size = config.get('training', {}).get('batch_size', 32)
    num_workers_pref = config.get('training', {}).get('num_workers', 'auto')
    
    device_config['batch_size'] = manager.get_memory_efficient_batch_size(device, base_batch_size)
    device_config['num_workers'] = manager.get_optimal_workers(device, num_workers_pref)
    
    # Print info
    manager.print_device_info(device)
    print(f"   Batch size: {device_config['batch_size']}")
    print(f"   Workers: {device_config['num_workers']}")
    print(f"   Pin memory: {device_config['pin_memory']}")
    
    return device, device_config


# Global device manager instance
device_manager = DeviceManager()


def get_device_manager() -> DeviceManager:
    """Get global device manager instance."""
    return device_manager
