#pragma once

#include "../core/tensor.h"
#include "../codegen/jit_compiler.h"
#include "../codegen/kernel_cache.h"
#include "../codegen/ir_generator.h"
#include "../codegen/llvm_module_builder.h"
#include <memory>

namespace mlc {

/// JIT-accelerated executor for tensor operations
/// Generates LLVM IR for operations, compiles them via JIT,
/// caches compiled kernels for reuse
class JITExecutor {
 public:
  JITExecutor();
  ~JITExecutor() = default;

  /// Execute tensor addition using JIT-compiled kernel
  void execute_add_jit(const Tensor& a, const Tensor& b, Tensor& output);

  /// Set optimization level for LLVM passes (0-3)
  void SetOptimizationLevel(int level);

  /// Get JIT optimization level
  int GetOptimizationLevel() const;

  /// Get kernel cache statistics
  struct CacheStats {
    size_t total_lookups;
    size_t cache_hits;
    size_t cache_misses;
    double hit_rate;
  };

  CacheStats GetCacheStats() const;
  void ResetCacheStats();
  void ClearCache();

  /// Get number of cached kernels
  size_t CachedKernelCount() const;

 private:
  std::unique_ptr<codegen::JITCompiler> jit_compiler_;
  std::unique_ptr<codegen::KernelCache> kernel_cache_;

  /// Generate a simple add kernel for given parameters
  std::unique_ptr<llvm::Module> GenerateAddKernelModule(
      size_t tensor_size, bool use_strides);

  /// Generate vectorized add kernel
  std::unique_ptr<llvm::Module> GenerateVectorizedAddKernelModule(
      size_t tensor_size, int vector_width);

  /// Execute compiled kernel on tensors
  void ExecuteCompiledKernel(
      const codegen::CompiledKernel& kernel,
      const Tensor& a, const Tensor& b, Tensor& output);
};

}  // namespace mlc
