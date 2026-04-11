#include <cassert>
#include <iostream>
#include <cmath>
#include <chrono>
#include "core/tensor.h"
#include "core/operation.h"
#include "runtime/executor.h"

using namespace mlc;

/// Test helper: compare vectors with tolerance
void assert_vectors_equal(const std::vector<float>& actual,
                           const std::vector<float>& expected,
                           float tolerance = 1e-5) {
  assert(actual.size() == expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    assert(std::abs(actual[i] - expected[i]) < tolerance);
  }
}

/// Test 1: Basic JIT compilation
void test_jit_basic_addition() {
  std::cout << "Test 1: Basic JIT addition (non-broadcast)..." << std::endl;
  
  std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> data_b = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
  std::vector<int> shape = {5};
  
  Tensor a(data_a, shape);
  Tensor b(data_b, shape);
  Tensor output(shape, Device::CPU);
  
  // Execute with JIT
  Executor::execute_add(a, b, output);
  
  std::vector<float> expected = {11.0f, 22.0f, 33.0f, 44.0f, 55.0f};
  assert_vectors_equal(output.to_vector(), expected);
  
  std::cout << "✓ Basic JIT addition test passed" << std::endl;
}

/// Test 2: 2D tensor addition with JIT
void test_jit_2d_addition() {
  std::cout << "Test 2: JIT 2D addition..." << std::endl;
  
  std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> data_b = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
  std::vector<int> shape = {2, 3};
  
  Tensor a(data_a, shape);
  Tensor b(data_b, shape);
  Tensor output(shape, Device::CPU);
  
  Executor::execute_add(a, b, output);
  
  std::vector<float> expected = {11.0f, 22.0f, 33.0f, 44.0f, 55.0f, 66.0f};
  assert_vectors_equal(output.to_vector(), expected);
  
  std::cout << "✓ 2D JIT addition test passed" << std::endl;
}

/// Test 3: Broadcasting with JIT
void test_jit_broadcast_addition() {
  std::cout << "Test 3: JIT addition with broadcasting..." << std::endl;
  
  std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> data_b = {10.0f, 20.0f, 30.0f};
  std::vector<int> shape_a = {2, 3};
  std::vector<int> shape_b = {1, 3};
  std::vector<int> output_shape = {2, 3};
  
  Tensor a(data_a, shape_a);
  Tensor b(data_b, shape_b);
  Tensor output(output_shape, Device::CPU);
  
  Executor::execute_add(a, b, output);
  
  std::vector<float> expected = {11.0f, 22.0f, 33.0f, 14.0f, 25.0f, 36.0f};
  assert_vectors_equal(output.to_vector(), expected);
  
  std::cout << "✓ Broadcasting JIT addition test passed" << std::endl;
}

/// Test 4: Kernel caching
void test_kernel_caching() {
  std::cout << "Test 4: Kernel caching..." << std::endl;
  
  std::vector<int> shape = {100};
  std::vector<float> data_a(100, 1.0f);
  std::vector<float> data_b(100, 2.0f);
  
  Tensor a(data_a, shape);
  Tensor b(data_b, shape);
  Tensor output(shape, Device::CPU);
  
  // First execution - cache miss
  Executor::execute_add(a, b, output);
  
  // Second execution - should hit cache
  Executor::execute_add(a, b, output);
  
  std::vector<float> expected(100, 3.0f);
  assert_vectors_equal(output.to_vector(), expected);
  
  std::cout << "✓ Kernel caching test passed" << std::endl;
}

/// Test 5: Fallback to naive path
void test_fallback_to_naive() {
  std::cout << "Test 5: Fallback to naive execution..." << std::endl;
  
  // Disable JIT to test fallback path
  bool original_jit_state = Executor::GetUseJIT();
  Executor::SetUseJIT(false);
  
  std::vector<float> data_a = {1.0f, 2.0f, 3.0f};
  std::vector<float> data_b = {4.0f, 5.0f, 6.0f};
  std::vector<int> shape = {3};
  
  Tensor a(data_a, shape);
  Tensor b(data_b, shape);
  Tensor output(shape, Device::CPU);
  
  Executor::execute_add(a, b, output);
  
  std::vector<float> expected = {5.0f, 7.0f, 9.0f};
  assert_vectors_equal(output.to_vector(), expected);
  
  // Restore JIT state
  Executor::SetUseJIT(original_jit_state);
  
  std::cout << "✓ Fallback to naive test passed" << std::endl;
}

/// Test 6: Large tensor performance
void test_large_tensor_jit() {
  std::cout << "Test 6: Large tensor JIT execution..." << std::endl;
  
  std::vector<int> shape = {1000000};  // 1M elements
  std::vector<float> data_a(1000000, 1.5f);
  std::vector<float> data_b(1000000, 2.5f);
  
  Tensor a(data_a, shape);
  Tensor b(data_b, shape);
  Tensor output(shape, Device::CPU);
  
  auto start = std::chrono::high_resolution_clock::now();
  Executor::execute_add(a, b, output);
  auto end = std::chrono::high_resolution_clock::now();
  
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "  Large tensor addition took: " << duration.count() << " µs" << std::endl;
  
  // Spot check first and last elements
  auto result = output.to_vector();
  assert(std::abs(result[0] - 4.0f) < 1e-5);
  assert(std::abs(result[999999] - 4.0f) < 1e-5);
  
  std::cout << "✓ Large tensor JIT test passed" << std::endl;
}

/// Test 7: Consistent results between JIT and naive
void test_jit_vs_naive_consistency() {
  std::cout << "Test 7: JIT vs naive consistency..." << std::endl;
  
  std::vector<float> data_a = {1.1f, 2.2f, 3.3f, 4.4f};
  std::vector<float> data_b = {5.5f, 6.6f, 7.7f, 8.8f};
  std::vector<int> shape = {4};
  
  Tensor a(data_a, shape);
  Tensor b(data_b, shape);
  
  // JIT path
  Tensor output_jit(shape, Device::CPU);
  Executor::SetUseJIT(true);
  Executor::execute_add(a, b, output_jit);
  auto result_jit = output_jit.to_vector();
  
  // Naive path
  Tensor output_naive(shape, Device::CPU);
  Executor::SetUseJIT(false);
  Executor::execute_add(a, b, output_naive);
  auto result_naive = output_naive.to_vector();
  
  // Compare results
  for (size_t i = 0; i < result_jit.size(); ++i) {
    assert(std::abs(result_jit[i] - result_naive[i]) < 1e-5);
  }
  
  std::cout << "✓ JIT vs naive consistency test passed" << std::endl;
}

/// Test 8: 3D tensor with JIT
void test_jit_3d_tensor() {
  std::cout << "Test 8: JIT 3D tensor addition..." << std::endl;
  
  std::vector<float> data_a(24);  // 2x3x4
  std::vector<float> data_b(24);
  for (int i = 0; i < 24; ++i) {
    data_a[i] = static_cast<float>(i + 1);
    data_b[i] = static_cast<float>(i + 100);
  }
  
  std::vector<int> shape = {2, 3, 4};
  Tensor a(data_a, shape);
  Tensor b(data_b, shape);
  Tensor output(shape, Device::CPU);
  
  Executor::execute_add(a, b, output);
  
  auto result = output.to_vector();
  for (int i = 0; i < 24; ++i) {
    float expected = (i + 1) + (i + 100);
    assert(std::abs(result[i] - expected) < 1e-5);
  }
  
  std::cout << "✓ 3D tensor JIT test passed" << std::endl;
}

int main() {
  try {
    std::cout << "\n=== JIT Executor Comprehensive Tests ===\n" << std::endl;
    
    test_jit_basic_addition();
    test_jit_2d_addition();
    test_jit_broadcast_addition();
    test_kernel_caching();
    test_fallback_to_naive();
    test_large_tensor_jit();
    test_jit_vs_naive_consistency();
    test_jit_3d_tensor();
    
    std::cout << "\n✅ All JIT tests passed!\n" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
    return 1;
  }
}
