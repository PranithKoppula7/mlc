"""MLC Tensor Compiler - CUDA-accelerated tensor operations

A lightweight tensor compiler that provides:
- Efficient tensor operations on CPU and GPU
- Lazy evaluation with computation graphs
- Automatic differentiation
- LLVM-based code generation for CPU
- CUDA support for GPU acceleration

Quick Start:
    import mlc
    a = mlc.Tensor([1.0, 2.0, 3.0], shape=[3])
    b = mlc.Tensor([4.0, 5.0, 6.0], shape=[3])
    c = a + b
    print(c)
"""

try:
    from .mlc_core import (Tensor, Device, DataType, zeros, ones, empty, full,
                            set_use_jit, get_use_jit, enable_jit, disable_jit)
except ImportError as e:
    raise ImportError(
        f"mlc_core module not found: {e}\n"
        "Please build the project with CMake:\n"
        "  cd /path/to/mlc\n"
        "  mkdir build && cd build\n"
        "  cmake ..\n"
        "  make\n"
        "Then add the lib directory to PYTHONPATH:\n"
        "  export PYTHONPATH=/path/to/mlc/build/lib:$PYTHONPATH"
    )

__version__ = "0.1.0"
__all__ = ["Tensor", "Device", "DataType", "zeros", "ones", "empty", "full", 
           "cpu", "cuda", "set_use_jit", "get_use_jit", "enable_jit", "disable_jit"]

# Convenience functions
def cpu(data=None, shape=None):
    """Create a CPU tensor
    
    Args:
        data: Optional initial data as list
        shape: Required shape as list [d0, d1, ...]
        
    Returns:
        Tensor on CPU device
    """
    if data is not None and shape is not None:
        return Tensor(data, shape=shape)
    elif shape is not None:
        return Tensor(shape=shape, device=Device.CPU)
    else:
        raise ValueError("Must provide shape (and optionally data)")

def cuda(data=None, shape=None):
    """Create a CUDA tensor (Phase 3)
    
    Args:
        data: Optional initial data as list
        shape: Required shape as list [d0, d1, ...]
        
    Returns:
        Tensor on CUDA device
    """
    if data is not None and shape is not None:
        return Tensor(data, shape=shape).to_device(Device.CUDA)
    elif shape is not None:
        return Tensor(shape=shape, device=Device.CUDA)
    else:
        raise ValueError("Must provide shape (and optionally data)")
