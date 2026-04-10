#!/usr/bin/env python3
"""
Example usage of MLC Tensor Compiler (Phase 1)

This demonstrates the API design. To run, first build the C++ library and
install pybind11 bindings:
    cd /home/pranith-dev/mlc
    mkdir build && cd build
    cmake .. && make
    
Then the bindings will be available as:
    import sys
    sys.path.insert(0, '/home/pranith-dev/mlc/build/lib')
    import mlc_core
"""

# Note: This is a demonstration of the intended API.
# The actual Python bindings require pybind11 to be installed.

def example_usage():
    """Demonstrates intended API usage"""
    
    # This is pseudo-code showing what the API would look like
    # when pybind11 bindings are available
    
    print("""
    ===== MLC Tensor Compiler - Phase 1 Example =====
    
    # Import the tensor library
    from mlc_core import Tensor, zeros, ones, Device
    
    # Create tensors
    a = Tensor([1.0, 2.0, 3.0], shape=[3])
    b = Tensor([4.0, 5.0, 6.0], shape=[3])
    
    # Perform operations (lazy evaluation)
    c = a + b  # Result is not computed yet
    
    # Compute and get result
    result = c.to_vector()  # [5.0, 7.0, 9.0]
    
    # 2D tensors
    x = Tensor([[1, 2], [3, 4]], shape=[2, 2])
    y = Tensor([[5, 6], [7, 8]], shape=[2, 2])
    z = x + y
    
    # Broadcasting support
    scalar = Tensor([10.0], shape=[1])
    broadcasted = x + scalar  # Broadcast to [2, 2]
    
    # Device placement (Phase 3)
    gpu_tensor = a.to_device(Device.CUDA)
    result = gpu_tensor.to_vector()  # Returns to CPU for inspection
    
    # Utility functions
    zeros_tensor = zeros(shape=[3, 4])
    ones_tensor = ones(shape=[2, 2])
    """)

if __name__ == "__main__":
    example_usage()
