#pragma once

#include "../core/tensor.h"

namespace mlc {

class Executor {
public:
    // Execute tensor addition
    static void execute_add(const Tensor& a, const Tensor& b, Tensor& output);
    
private:
    // CPU execution paths
    static void execute_add_cpu_naive(const Tensor& a, const Tensor& b, Tensor& output);
    static void execute_add_cpu_vectorized(const Tensor& a, const Tensor& b, Tensor& output);
};

}  // namespace mlc
