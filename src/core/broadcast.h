#pragma once

#include <vector>
#include <algorithm>

namespace mlc {

class BroadcastResolver {
public:
    // Compute output shape when broadcasting two tensors
    static std::vector<int> compute_output_shape(
        const std::vector<int>& shape_a,
        const std::vector<int>& shape_b);
    
    // Compute strides for efficient broadcasting
    static std::vector<int> compute_strides_for_broadcast(
        const std::vector<int>& original_shape,
        const std::vector<int>& broadcast_shape);
    
    // Check if two shapes are broadcastable
    static bool is_broadcastable(
        const std::vector<int>& shape_a,
        const std::vector<int>& shape_b);
};

}  // namespace mlc
