#include "executor.h"
#include "jit_executor.h"
#include "../core/broadcast.h"
#include <stdexcept>
#include <cstring>
#include <algorithm>

namespace mlc {

bool Executor::use_jit_ = true;

std::unique_ptr<JITExecutor>& Executor::GetJITExecutor() {
    static std::unique_ptr<JITExecutor> jit_executor = std::make_unique<JITExecutor>();
    return jit_executor;
}

void Executor::SetUseJIT(bool use_jit) {
    use_jit_ = use_jit;
}

bool Executor::GetUseJIT() {
    return use_jit_;
}

void Executor::execute_add(const Tensor& a, const Tensor& b, Tensor& output) {
    if (a.device() != Device::CPU || b.device() != Device::CPU) {
        throw std::invalid_argument("Phase 2 only supports CPU execution");
    }
    
    // Check broadcast compatibility
    if (!BroadcastResolver::is_broadcastable(a.shape(), b.shape())) {
        throw std::invalid_argument("Shapes are not broadcastable");
    }
    
    // Use JIT if enabled, otherwise fall back to naive path
    if (use_jit_) {
        try {
            GetJITExecutor()->execute_add_jit(a, b, output);
            return;
        } catch (const std::exception& e) {
            // Fall back to naive path on error
            // In production, you might want to log this
            execute_add_cpu_naive(a, b, output);
        }
    } else {
        execute_add_cpu_naive(a, b, output);
    }
}

void Executor::execute_add_cpu_naive(const Tensor& a, const Tensor& b, Tensor& output) {
    // Get broadcast information
    std::vector<int> output_shape = output.shape();
    std::vector<int> a_broadcast_strides = 
        BroadcastResolver::compute_strides_for_broadcast(a.shape(), output_shape);
    std::vector<int> b_broadcast_strides = 
        BroadcastResolver::compute_strides_for_broadcast(b.shape(), output_shape);
    
    size_t total_size = output.size();
    
    // Simple element-wise addition with broadcasting
    for (size_t i = 0; i < total_size; ++i) {
        // Compute linear index to multi-dimensional index
        size_t idx_a = 0;
        size_t idx_b = 0;
        size_t remaining = i;
        
        for (int d = output_shape.size() - 1; d >= 0; --d) {
            size_t coord = remaining % output_shape[d];
            remaining /= output_shape[d];
            
            if (d < static_cast<int>(a_broadcast_strides.size())) {
                idx_a += coord * a_broadcast_strides[d];
            }
            if (d < static_cast<int>(b_broadcast_strides.size())) {
                idx_b += coord * b_broadcast_strides[d];
            }
        }
        
        output.data()[i] = a.data()[idx_a] + b.data()[idx_b];
    }
}

void Executor::execute_add_cpu_vectorized(const Tensor& a, const Tensor& b, Tensor& output) {
    // Phase 2: Vectorized SIMD implementation
    // For now, fall back to naive
    execute_add_cpu_naive(a, b, output);
}

}  // namespace mlc
