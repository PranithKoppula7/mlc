#pragma once

#include "../core/tensor.h"
#include <memory>

namespace mlc {

class JITExecutor;

class Executor {
public:
    // Execute tensor addition using JIT by default
    // Falls back to naive path if JIT is disabled or encounters errors
    static void execute_add(const Tensor& a, const Tensor& b, Tensor& output);
    
    // Enable/disable JIT execution (JIT is enabled by default)
    static void SetUseJIT(bool use_jit);
    static bool GetUseJIT();
    
private:
    // CPU execution paths
    static void execute_add_cpu_naive(const Tensor& a, const Tensor& b, Tensor& output);
    static void execute_add_cpu_vectorized(const Tensor& a, const Tensor& b, Tensor& output);
    
    // Global JIT executor instance
    static std::unique_ptr<JITExecutor>& GetJITExecutor();
    static bool use_jit_;
};

}  // namespace mlc
