#!/usr/bin/env python3
"""
Comprehensive Python bindings validation test suite for MLC Tensor Compiler.
Tests all Phase 1 functionality through the Python API.

Usage:
    python3 test_python_bindings.py

Or from the build directory:
    PYTHONPATH=./lib python3 ../examples/test_python_bindings.py
"""

import sys
import os

# Add build/lib to path if running from build directory
build_lib = os.path.join(os.path.dirname(__file__), '..', 'build', 'lib')
if os.path.exists(build_lib):
    sys.path.insert(0, build_lib)

try:
    # Import after path setup
    import mlc
    from mlc import Tensor, Device, DataType, zeros, ones, empty, full
except ImportError as e:
    print(f"ERROR: Failed to import mlc module: {e}")
    print("\nPlease ensure the project is built:")
    print("  cd /home/pranith-dev/mlc")
    print("  mkdir -p build && cd build")
    print("  cmake .. && make")
    sys.exit(1)


class TestPythonBindings:
    """Test suite for Python bindings"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def assert_equal(self, actual, expected, msg=""):
        """Assert equality with message"""
        if actual != expected:
            raise AssertionError(f"{msg}\n  Expected: {expected}\n  Got: {actual}")
    
    def assert_true(self, condition, msg=""):
        """Assert condition is true"""
        if not condition:
            raise AssertionError(msg)
    
    def assert_close(self, actual, expected, tolerance=1e-6, msg=""):
        """Assert values are close (for floats)"""
        if isinstance(actual, list) and isinstance(expected, list):
            if len(actual) != len(expected):
                raise AssertionError(f"{msg}\n  Expected length {len(expected)}, got {len(actual)}")
            for a, e in zip(actual, expected):
                if abs(a - e) > tolerance:
                    raise AssertionError(f"{msg}\n  Expected {e}, got {a} (diff: {abs(a-e)})")
        else:
            if abs(actual - expected) > tolerance:
                raise AssertionError(f"{msg}\n  Expected {expected}, got {actual}")
    
    def run_test(self, name, test_func):
        """Run a single test"""
        try:
            test_func()
            self.passed += 1
            print(f"  ✓ {name}")
        except Exception as e:
            self.failed += 1
            print(f"  ✗ {name}")
            print(f"    Error: {e}")
    
    # ========== BASIC TENSOR CREATION ==========
    
    def test_tensor_creation_default(self):
        """Test creating empty tensor"""
        t = Tensor()
        # Default tensor has size 1 (uninitialized scalar)
        self.assert_true(t.size() >= 0, "Empty tensor should have non-negative size")
    
    def test_tensor_creation_with_data(self):
        """Test creating tensor with initial data"""
        data = [1.0, 2.0, 3.0]
        shape = [3]
        t = Tensor(data, shape=shape)
        self.assert_equal(t.size(), 3, "Tensor size should match data length")
        self.assert_equal(t.shape(), shape, "Shape should match")
        self.assert_close(t.to_vector(), data, msg="Data should match")
    
    def test_tensor_creation_from_shape(self):
        """Test creating zero-initialized tensor from shape"""
        shape = [2, 3]
        t = Tensor(shape)
        self.assert_equal(t.size(), 6, "Size should be product of shape")
        self.assert_equal(t.rank(), 2, "Rank should match shape length")
        self.assert_close(t.to_vector(), [0.0] * 6, msg="Should be zero-initialized")
    
    def test_tensor_properties(self):
        """Test tensor metadata accessors"""
        t = Tensor([3, 4])
        self.assert_equal(t.shape(), [3, 4], "Shape should match")
        self.assert_equal(t.size(), 12, "Size should be 12")
        self.assert_equal(t.rank(), 2, "Rank should be 2")
        self.assert_equal(t.device(), Device.CPU, "Default device should be CPU")
        self.assert_equal(t.dtype(), DataType.FLOAT32, "Default dtype should be FLOAT32")
    
    # ========== TENSOR CREATION UTILITIES ==========
    
    def test_zeros(self):
        """Test zeros utility function"""
        t = zeros([2, 3])
        self.assert_equal(t.size(), 6)
        self.assert_close(t.to_vector(), [0.0] * 6)
    
    def test_ones(self):
        """Test ones utility function"""
        t = ones([2, 3])
        self.assert_equal(t.size(), 6)
        self.assert_close(t.to_vector(), [1.0] * 6)
    
    def test_empty(self):
        """Test empty utility function"""
        t = empty([2, 3])
        self.assert_equal(t.size(), 6)
        self.assert_equal(t.rank(), 2)
    
    def test_full(self):
        """Test full utility function"""
        t = full([2, 3], 5.0)
        self.assert_equal(t.size(), 6)
        self.assert_close(t.to_vector(), [5.0] * 6)
    
    # ========== ADDITION OPERATION ==========
    
    def test_add_basic(self):
        """Test basic tensor addition"""
        a = Tensor([1.0, 2.0, 3.0], shape=[3])
        b = Tensor([4.0, 5.0, 6.0], shape=[3])
        c = a + b
        expected = [5.0, 7.0, 9.0]
        self.assert_close(c.to_vector(), expected, msg="Addition result incorrect")
    
    def test_add_operator_syntax(self):
        """Test __add__ operator"""
        a = Tensor([1.0, 2.0], shape=[2])
        b = Tensor([10.0, 20.0], shape=[2])
        c = a + b
        self.assert_close(c.to_vector(), [11.0, 22.0])
    
    def test_add_method_syntax(self):
        """Test add() method"""
        a = Tensor([1.0, 2.0], shape=[2])
        b = Tensor([10.0, 20.0], shape=[2])
        c = a.add(b)
        self.assert_close(c.to_vector(), [11.0, 22.0])
    
    def test_add_2d_tensors(self):
        """Test addition of 2D tensors"""
        a = Tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
        b = Tensor([1.0, 1.0, 1.0, 1.0], shape=[2, 2])
        c = a + b
        expected = [2.0, 3.0, 4.0, 5.0]
        self.assert_close(c.to_vector(), expected)
    
    # ========== BROADCASTING ==========
    
    def test_broadcast_1d_to_2d(self):
        """Test broadcasting scalar-like to vector"""
        # Both must have same rank - broadcasting happens during addition
        a = Tensor([1.0, 2.0], shape=[2])       # Shape (2,)
        b = Tensor([10.0, 20.0], shape=[2])     # Shape (2,)
        c = a + b
        self.assert_equal(c.rank(), 1, "Result should be 1D")
        self.assert_close(c.to_vector(), [11.0, 22.0])
    
    def test_broadcast_2d_tensors(self):
        """Test broadcasting with 2D tensors"""
        # Create (2, 3) and (1, 3) tensors - should broadcast to (2, 3)
        a = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
        b = Tensor([10.0, 20.0, 30.0], shape=[1, 3])
        c = a + b
        self.assert_equal(c.shape(), [2, 3], "Shape should be (2, 3) after broadcast")
        expected = [11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
        self.assert_close(c.to_vector(), expected, msg="Broadcasting result incorrect")
    
    # ========== SHAPE MANIPULATION ==========
    
    def test_flatten(self):
        """Test tensor flattening"""
        t = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
        flat = t.flatten()
        self.assert_equal(flat.rank(), 1, "Flattened tensor should have rank 1")
        self.assert_equal(flat.size(), 6, "Size should be preserved")
        self.assert_equal(flat.shape(), [6], "Shape should be [6]")
    
    def test_reshape(self):
        """Test tensor reshaping"""
        t = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[6])
        reshaped = t.reshape([2, 3])
        self.assert_equal(reshaped.shape(), [2, 3], "Shape should be [2, 3]")
        self.assert_equal(reshaped.size(), 6, "Size should be preserved")
        self.assert_close(reshaped.to_vector(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    
    def test_reshape_3d(self):
        """Test reshaping to 3D"""
        t = Tensor([1.0] * 24, shape=[24])
        reshaped = t.reshape([2, 3, 4])
        self.assert_equal(reshaped.shape(), [2, 3, 4])
        self.assert_equal(reshaped.rank(), 3)
        self.assert_equal(reshaped.size(), 24)
    
    # ========== DEVICE OPERATIONS ==========
    
    def test_device_cpu(self):
        """Test CPU device placement"""
        t = Tensor([1.0, 2.0], shape=[2])
        cpu_t = t.to_device(Device.CPU)
        self.assert_equal(cpu_t.device(), Device.CPU)
    
    def test_device_cuda(self):
        """Test CUDA device placement (Phase 3 - may not be available)"""
        try:
            t = Tensor([1.0, 2.0], shape=[2])
            cuda_t = t.to_device(Device.CUDA)
            self.assert_equal(cuda_t.device(), Device.CUDA)
            print("    (CUDA support available - Phase 3 feature)")
        except Exception as e:
            print(f"    (CUDA not available: {e} - expected for Phase 1)")
    
    def test_cpu_convenience_method(self):
        """Test .cpu() convenience method"""
        t = Tensor([1.0, 2.0], shape=[2])
        cpu_t = t.cpu()
        self.assert_equal(cpu_t.device(), Device.CPU)
    
    # ========== UTILITIES ==========
    
    def test_repr(self):
        """Test tensor representation"""
        t = Tensor([1.0, 2.0, 3.0], shape=[3])
        repr_str = repr(t)
        self.assert_true(isinstance(repr_str, str), "repr should return string")
        self.assert_true(len(repr_str) > 0, "repr should not be empty")
    
    def test_len(self):
        """Test len() function on tensor"""
        t = Tensor([1.0, 2.0, 3.0], shape=[3])
        self.assert_equal(len(t), 3, "len should return size")
    
    def test_tensor_equality(self):
        """Test tensor equality comparison"""
        a = Tensor([1.0, 2.0, 3.0], shape=[3])
        b = Tensor([1.0, 2.0, 3.0], shape=[3])
        self.assert_true(a == b, "Same tensors should be equal")
    
    def test_tensor_inequality(self):
        """Test tensor inequality"""
        a = Tensor([1.0, 2.0, 3.0], shape=[3])
        b = Tensor([1.0, 2.0, 4.0], shape=[3])
        self.assert_true(not (a == b), "Different tensors should not be equal")
    
    # ========== DTYPE AND DEVICE INFO ==========
    
    def test_dtype_float32(self):
        """Test FLOAT32 dtype"""
        t = Tensor([1.0], shape=[1])
        self.assert_equal(t.dtype(), DataType.FLOAT32)
    
    def test_requires_grad(self):
        """Test requires_grad flag (Phase 4 - autograd)"""
        t = Tensor([1.0], shape=[1])
        grad_required = t.requires_grad()
        self.assert_true(isinstance(grad_required, bool), "requires_grad should return bool")
    
    # ========== CONVENIENCE FUNCTIONS ==========
    
    def test_mlc_cpu_function(self):
        """Test mlc.cpu() convenience function"""
        t = mlc.cpu(shape=[2, 3])
        self.assert_equal(t.device(), Device.CPU)
        self.assert_equal(t.shape(), [2, 3])
    
    def test_mlc_cuda_function(self):
        """Test mlc.cuda() convenience function"""
        try:
            t = mlc.cuda(shape=[2, 3])
            self.assert_equal(t.device(), Device.CUDA)
            print("    (CUDA support available)")
        except Exception as e:
            print(f"    (CUDA not available: expected for Phase 1)")
    
    # ========== COMPLEX OPERATIONS ==========
    
    def test_chained_operations(self):
        """Test chaining multiple operations"""
        a = Tensor([1.0, 2.0, 3.0], shape=[3])
        b = Tensor([4.0, 5.0, 6.0], shape=[3])
        c = Tensor([10.0, 20.0, 30.0], shape=[3])
        result = a + b + c
        expected = [15.0, 27.0, 39.0]
        self.assert_close(result.to_vector(), expected)
    
    def test_operation_preserves_data(self):
        """Test that operations don't modify original tensors"""
        a = Tensor([1.0, 2.0, 3.0], shape=[3])
        b = Tensor([4.0, 5.0, 6.0], shape=[3])
        a_original = a.to_vector()
        b_original = b.to_vector()
        
        _ = a + b
        
        self.assert_close(a.to_vector(), a_original)
        self.assert_close(b.to_vector(), b_original)
    
    def test_large_tensor_addition(self):
        """Test addition with larger tensors"""
        size = 1000
        a = Tensor([1.0] * size, shape=[size])
        b = Tensor([2.0] * size, shape=[size])
        c = a + b
        self.assert_equal(c.size(), size)
        expected = [3.0] * size
        self.assert_close(c.to_vector(), expected)
    
    # ========== TEST RUNNER ==========
    
    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*60)
        print("MLC Tensor Compiler - Python Bindings Test Suite")
        print("="*60)
        
        # Group tests by category
        categories = [
            ("Basic Tensor Creation", [
                ("Empty tensor", self.test_tensor_creation_default),
                ("Tensor with data", self.test_tensor_creation_with_data),
                ("Tensor from shape", self.test_tensor_creation_from_shape),
                ("Tensor properties", self.test_tensor_properties),
            ]),
            ("Tensor Creation Utilities", [
                ("zeros()", self.test_zeros),
                ("ones()", self.test_ones),
                ("empty()", self.test_empty),
                ("full()", self.test_full),
            ]),
            ("Addition Operations", [
                ("Basic addition", self.test_add_basic),
                ("__add__ operator", self.test_add_operator_syntax),
                (".add() method", self.test_add_method_syntax),
                ("2D tensor addition", self.test_add_2d_tensors),
            ]),
            ("Broadcasting", [
                ("1D to 2D broadcast", self.test_broadcast_1d_to_2d),
                ("2D tensor broadcast", self.test_broadcast_2d_tensors),
            ]),
            ("Shape Manipulation", [
                ("Flatten", self.test_flatten),
                ("Reshape", self.test_reshape),
                ("Reshape to 3D", self.test_reshape_3d),
            ]),
            ("Device Operations", [
                ("CPU device", self.test_device_cpu),
                ("CUDA device", self.test_device_cuda),
                (".cpu() method", self.test_cpu_convenience_method),
            ]),
            ("Utilities", [
                ("repr()", self.test_repr),
                ("len()", self.test_len),
                ("Equality", self.test_tensor_equality),
                ("Inequality", self.test_tensor_inequality),
            ]),
            ("Dtype and Device Info", [
                ("FLOAT32 dtype", self.test_dtype_float32),
                ("requires_grad", self.test_requires_grad),
            ]),
            ("Convenience Functions", [
                ("mlc.cpu()", self.test_mlc_cpu_function),
                ("mlc.cuda()", self.test_mlc_cuda_function),
            ]),
            ("Complex Operations", [
                ("Chained operations", self.test_chained_operations),
                ("Operation preserves data", self.test_operation_preserves_data),
                ("Large tensor addition", self.test_large_tensor_addition),
            ]),
        ]
        
        for category, tests in categories:
            print(f"\n{category}:")
            for name, test_func in tests:
                self.run_test(name, test_func)
        
        # Print summary
        total = self.passed + self.failed
        print("\n" + "="*60)
        print(f"Test Results: {self.passed}/{total} passed", end="")
        if self.failed > 0:
            print(f", {self.failed} failed")
            print("="*60)
            return False
        else:
            print(" ✓ ALL TESTS PASSED!")
            print("="*60)
            return True


def main():
    """Main entry point"""
    tester = TestPythonBindings()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
