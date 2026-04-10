#include <cassert>
#include <iostream>
#include <cmath>
#include "core/tensor.h"
#include "core/operation.h"

using namespace mlc;

void test_simple_add() {
    std::cout << "Testing simple addition..." << std::endl;
    
    std::vector<float> data_a = {1.0f, 2.0f, 3.0f};
    std::vector<float> data_b = {4.0f, 5.0f, 6.0f};
    std::vector<int> shape = {3};
    
    Tensor a(data_a, shape);
    Tensor b(data_b, shape);
    
    AddOp add_op;
    auto result = add_op.forward({a, b});
    
    assert(result.size() == 1);
    Tensor output = result[0];
    
    auto output_vec = output.to_vector();
    std::vector<float> expected = {5.0f, 7.0f, 9.0f};
    
    for (int i = 0; i < 3; ++i) {
        assert(std::abs(output_vec[i] - expected[i]) < 1e-6);
    }
    
    std::cout << "✓ Simple addition test passed" << std::endl;
}

void test_2d_add() {
    std::cout << "Testing 2D addition..." << std::endl;
    
    std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data_b = {5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<int> shape = {2, 2};
    
    Tensor a(data_a, shape);
    Tensor b(data_b, shape);
    
    AddOp add_op;
    auto result = add_op.forward({a, b});
    
    Tensor output = result[0];
    auto output_vec = output.to_vector();
    std::vector<float> expected = {6.0f, 8.0f, 10.0f, 12.0f};
    
    for (int i = 0; i < 4; ++i) {
        assert(std::abs(output_vec[i] - expected[i]) < 1e-6);
    }
    
    std::cout << "✓ 2D addition test passed" << std::endl;
}

void test_broadcast_add_scalar() {
    std::cout << "Testing addition with scalar broadcast..." << std::endl;
    
    std::vector<float> data_a = {1.0f, 2.0f, 3.0f};
    std::vector<float> data_b = {10.0f};
    std::vector<int> shape_a = {3};
    std::vector<int> shape_b = {1};
    
    Tensor a(data_a, shape_a);
    Tensor b(data_b, shape_b);
    
    AddOp add_op;
    auto result = add_op.forward({a, b});
    
    Tensor output = result[0];
    assert(output.shape()[0] == 3);
    
    auto output_vec = output.to_vector();
    std::vector<float> expected = {11.0f, 12.0f, 13.0f};
    
    for (int i = 0; i < 3; ++i) {
        assert(std::abs(output_vec[i] - expected[i]) < 1e-6);
    }
    
    std::cout << "✓ Broadcast scalar addition test passed" << std::endl;
}

void test_broadcast_add_2d() {
    std::cout << "Testing 2D broadcast addition..." << std::endl;
    
    // (2, 3) + (1, 3) = (2, 3)
    std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> data_b = {10.0f, 20.0f, 30.0f};
    std::vector<int> shape_a = {2, 3};
    std::vector<int> shape_b = {1, 3};
    
    Tensor a(data_a, shape_a);
    Tensor b(data_b, shape_b);
    
    AddOp add_op;
    auto result = add_op.forward({a, b});
    
    Tensor output = result[0];
    assert(output.shape()[0] == 2);
    assert(output.shape()[1] == 3);
    
    auto output_vec = output.to_vector();
    std::vector<float> expected = {11.0f, 22.0f, 33.0f, 14.0f, 25.0f, 36.0f};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        assert(std::abs(output_vec[i] - expected[i]) < 1e-6);
    }
    
    std::cout << "✓ 2D broadcast addition test passed" << std::endl;
}

void test_add_output_shape() {
    std::cout << "Testing AddOp output shape computation..." << std::endl;
    
    Tensor a({2, 3}, Device::CPU);
    Tensor b({1, 3}, Device::CPU);
    
    AddOp add_op;
    auto output_shape = add_op.compute_output_shape({a, b});
    
    assert(output_shape.size() == 2);
    assert(output_shape[0] == 2);
    assert(output_shape[1] == 3);
    
    std::cout << "✓ AddOp output shape test passed" << std::endl;
}

int main() {
    try {
        test_simple_add();
        test_2d_add();
        test_broadcast_add_scalar();
        test_broadcast_add_2d();
        test_add_output_shape();
        
        std::cout << "\n✅ All Addition tests passed!\n" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
}
