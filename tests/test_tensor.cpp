#include <cassert>
#include <iostream>
#include <cmath>
#include "core/tensor.h"

using namespace mlc;

void test_tensor_creation() {
    std::cout << "Testing Tensor creation..." << std::endl;
    
    // Test basic creation
    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    std::vector<int> shape = {3};
    Tensor t(data, shape);
    
    assert(t.size() == 3);
    assert(t.rank() == 1);
    assert(t.shape()[0] == 3);
    assert(t.device() == Device::CPU);
    
    // Verify data
    auto t_vec = t.to_vector();
    for (int i = 0; i < 3; ++i) {
        assert(std::abs(t_vec[i] - data[i]) < 1e-6);
    }
    
    std::cout << "✓ Tensor creation test passed" << std::endl;
}

void test_tensor_2d() {
    std::cout << "Testing 2D Tensor..." << std::endl;
    
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int> shape = {2, 2};
    Tensor t(data, shape);
    
    assert(t.size() == 4);
    assert(t.rank() == 2);
    assert(t.shape()[0] == 2);
    assert(t.shape()[1] == 2);
    
    // Check strides
    auto strides = t.strides();
    assert(strides[0] == 2);
    assert(strides[1] == 1);
    
    std::cout << "✓ 2D Tensor test passed" << std::endl;
}

void test_reshape() {
    std::cout << "Testing reshape..." << std::endl;
    
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int> shape = {4};
    Tensor t(data, shape);
    
    Tensor reshaped = t.reshape({2, 2});
    assert(reshaped.size() == 4);
    assert(reshaped.shape()[0] == 2);
    assert(reshaped.shape()[1] == 2);
    
    std::cout << "✓ Reshape test passed" << std::endl;
}

void test_flatten() {
    std::cout << "Testing flatten..." << std::endl;
    
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int> shape = {2, 2};
    Tensor t(data, shape);
    
    Tensor flat = t.flatten();
    assert(flat.size() == 4);
    assert(flat.rank() == 1);
    assert(flat.shape()[0] == 4);
    
    std::cout << "✓ Flatten test passed" << std::endl;
}

void test_device_placement() {
    std::cout << "Testing device placement..." << std::endl;
    
    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    std::vector<int> shape = {3};
    Tensor t_cpu(data, shape);
    
    assert(t_cpu.device() == Device::CPU);
    
    // Note: GPU placement will be tested in Phase 3
    
    std::cout << "✓ Device placement test passed" << std::endl;
}

int main() {
    try {
        test_tensor_creation();
        test_tensor_2d();
        test_reshape();
        test_flatten();
        test_device_placement();
        
        std::cout << "\n✅ All Tensor tests passed!\n" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
}
