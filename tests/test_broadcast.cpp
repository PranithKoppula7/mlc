#include <cassert>
#include <iostream>
#include <vector>
#include "core/broadcast.h"

using namespace mlc;

void test_same_shape_broadcast() {
    std::cout << "Testing broadcast with same shapes..." << std::endl;
    
    std::vector<int> shape_a = {3};
    std::vector<int> shape_b = {3};
    
    auto output_shape = BroadcastResolver::compute_output_shape(shape_a, shape_b);
    assert(output_shape.size() == 1);
    assert(output_shape[0] == 3);
    
    assert(BroadcastResolver::is_broadcastable(shape_a, shape_b));
    
    std::cout << "✓ Same shape broadcast test passed" << std::endl;
}

void test_scalar_broadcast() {
    std::cout << "Testing broadcast with scalars (1D)..." << std::endl;
    
    std::vector<int> shape_a = {3};
    std::vector<int> shape_b = {1};
    
    auto output_shape = BroadcastResolver::compute_output_shape(shape_a, shape_b);
    assert(output_shape.size() == 1);
    assert(output_shape[0] == 3);
    
    assert(BroadcastResolver::is_broadcastable(shape_a, shape_b));
    assert(BroadcastResolver::is_broadcastable(shape_b, shape_a));
    
    std::cout << "✓ Scalar broadcast test passed" << std::endl;
}

void test_2d_broadcast() {
    std::cout << "Testing 2D broadcast..." << std::endl;
    
    // (2, 3) + (1, 3) = (2, 3)
    std::vector<int> shape_a = {2, 3};
    std::vector<int> shape_b = {1, 3};
    
    auto output_shape = BroadcastResolver::compute_output_shape(shape_a, shape_b);
    assert(output_shape.size() == 2);
    assert(output_shape[0] == 2);
    assert(output_shape[1] == 3);
    
    // (2, 3) + (1,) should broadcast to (2, 3)
    std::vector<int> shape_c = {1};
    auto output_shape_2 = BroadcastResolver::compute_output_shape(shape_a, shape_c);
    assert(output_shape_2.size() == 2);
    assert(output_shape_2[0] == 2);
    assert(output_shape_2[1] == 3);
    
    std::cout << "✓ 2D broadcast test passed" << std::endl;
}

void test_incompatible_broadcast() {
    std::cout << "Testing incompatible broadcast (should fail)..." << std::endl;
    
    std::vector<int> shape_a = {2, 3};
    std::vector<int> shape_b = {4, 3};
    
    assert(!BroadcastResolver::is_broadcastable(shape_a, shape_b));
    
    try {
        BroadcastResolver::compute_output_shape(shape_a, shape_b);
        assert(false);  // Should have thrown
    } catch (const std::invalid_argument&) {
        // Expected
    }
    
    std::cout << "✓ Incompatible broadcast test passed" << std::endl;
}

void test_strides_computation() {
    std::cout << "Testing stride computation for broadcasting..." << std::endl;
    
    // Original shape (1, 3), broadcast to (2, 3)
    std::vector<int> original = {1, 3};
    std::vector<int> broadcast = {2, 3};
    
    auto strides = BroadcastResolver::compute_strides_for_broadcast(original, broadcast);
    assert(strides.size() == 2);
    assert(strides[0] == 0);  // Dimension is broadcast, stride is 0
    assert(strides[1] == 1);  // Dimension is kept
    
    std::cout << "✓ Stride computation test passed" << std::endl;
}

void test_3d_broadcast() {
    std::cout << "Testing 3D broadcast..." << std::endl;
    
    // (1, 2, 3) + (4, 1, 3) = (4, 2, 3)
    std::vector<int> shape_a = {1, 2, 3};
    std::vector<int> shape_b = {4, 1, 3};
    
    auto output_shape = BroadcastResolver::compute_output_shape(shape_a, shape_b);
    assert(output_shape.size() == 3);
    assert(output_shape[0] == 4);
    assert(output_shape[1] == 2);
    assert(output_shape[2] == 3);
    
    std::cout << "✓ 3D broadcast test passed" << std::endl;
}

int main() {
    try {
        test_same_shape_broadcast();
        test_scalar_broadcast();
        test_2d_broadcast();
        test_incompatible_broadcast();
        test_strides_computation();
        test_3d_broadcast();
        
        std::cout << "\n✅ All Broadcast tests passed!\n" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
}
