#include "broadcast.h"
#include <stdexcept>
#include <numeric>

namespace mlc {

std::vector<int> BroadcastResolver::compute_output_shape(
    const std::vector<int>& shape_a,
    const std::vector<int>& shape_b) {
    
    // Align shapes to same rank by prepending 1s
    int rank_a = shape_a.size();
    int rank_b = shape_b.size();
    int max_rank = std::max(rank_a, rank_b);
    
    std::vector<int> aligned_a(max_rank, 1);
    std::vector<int> aligned_b(max_rank, 1);
    
    std::copy(shape_a.begin(), shape_a.end(), 
              aligned_a.begin() + (max_rank - rank_a));
    std::copy(shape_b.begin(), shape_b.end(), 
              aligned_b.begin() + (max_rank - rank_b));
    
    // Compute output shape
    std::vector<int> output_shape(max_rank);
    for (int i = 0; i < max_rank; ++i) {
        int dim_a = aligned_a[i];
        int dim_b = aligned_b[i];
        
        // Broadcasting rules: dimensions must be compatible
        if (dim_a == 1) {
            output_shape[i] = dim_b;
        } else if (dim_b == 1) {
            output_shape[i] = dim_a;
        } else if (dim_a == dim_b) {
            output_shape[i] = dim_a;
        } else {
            throw std::invalid_argument(
                "Shapes are not broadcastable: dimension mismatch");
        }
    }
    
    return output_shape;
}

std::vector<int> BroadcastResolver::compute_strides_for_broadcast(
    const std::vector<int>& original_shape,
    const std::vector<int>& broadcast_shape) {
    
    if (original_shape.size() != broadcast_shape.size()) {
        throw std::invalid_argument("Shape ranks must match after alignment");
    }
    
    int rank = original_shape.size();
    std::vector<int> strides(rank, 0);
    
    // If a dimension is 1 in original but > 1 in broadcast, stride is 0
    // (broadcast that dimension)
    int stride = 1;
    for (int i = rank - 1; i >= 0; --i) {
        if (original_shape[i] == broadcast_shape[i]) {
            strides[i] = stride;
            stride *= original_shape[i];
        } else if (original_shape[i] == 1) {
            strides[i] = 0;  // Broadcast this dimension
        } else {
            throw std::invalid_argument(
                "Cannot broadcast: incompatible dimensions");
        }
    }
    
    return strides;
}

bool BroadcastResolver::is_broadcastable(
    const std::vector<int>& shape_a,
    const std::vector<int>& shape_b) {
    
    int rank_a = shape_a.size();
    int rank_b = shape_b.size();
    int max_rank = std::max(rank_a, rank_b);
    
    std::vector<int> aligned_a(max_rank, 1);
    std::vector<int> aligned_b(max_rank, 1);
    
    std::copy(shape_a.begin(), shape_a.end(), 
              aligned_a.begin() + (max_rank - rank_a));
    std::copy(shape_b.begin(), shape_b.end(), 
              aligned_b.begin() + (max_rank - rank_b));
    
    for (int i = 0; i < max_rank; ++i) {
        int dim_a = aligned_a[i];
        int dim_b = aligned_b[i];
        
        if (dim_a != 1 && dim_b != 1 && dim_a != dim_b) {
            return false;
        }
    }
    
    return true;
}

}  // namespace mlc
