#include "jit_executor.h"
#include "../core/broadcast.h"
#include <stdexcept>

namespace mlc {

JITExecutor::JITExecutor()
    : jit_compiler_(std::make_unique<codegen::JITCompiler>()),
      kernel_cache_(std::make_unique<codegen::KernelCache>()) {}

std::unique_ptr<llvm::Module> JITExecutor::GenerateAddKernelModule(
    size_t tensor_size, bool use_strides) {
  auto module_builder = std::make_unique<codegen::LLVMModuleBuilder>("add_kernel");
  auto ir_gen = std::make_unique<codegen::IRGenerator>(module_builder.get());

  // Generate appropriate kernel based on parameters
  if (use_strides) {
    ir_gen->GenerateAddKernelWithStrides();
  } else {
    ir_gen->GenerateAddKernel();
  }

  // Return ownership of the module
  auto module = std::unique_ptr<llvm::Module>(module_builder->GetModule());
  module_builder.release();  // Module is now owned by returned unique_ptr
  return module;
}

std::unique_ptr<llvm::Module> JITExecutor::GenerateVectorizedAddKernelModule(
    size_t tensor_size, int vector_width) {
  auto module_builder = std::make_unique<codegen::LLVMModuleBuilder>("vec_add_kernel");
  auto ir_gen = std::make_unique<codegen::IRGenerator>(module_builder.get());

  ir_gen->GenerateVectorizedAddKernel(vector_width);

  auto module = std::unique_ptr<llvm::Module>(module_builder->GetModule());
  module_builder.release();
  return module;
}

void JITExecutor::execute_add_jit(const Tensor& a, const Tensor& b, Tensor& output) {
  if (a.device() != Device::CPU || b.device() != Device::CPU) {
    throw std::invalid_argument("JITExecutor: Only CPU device is supported");
  }

  // Check broadcast compatibility
  if (!BroadcastResolver::is_broadcastable(a.shape(), b.shape())) {
    throw std::invalid_argument("Shapes are not broadcastable");
  }

  // Determine if we need strided access (for broadcasting)
  bool needs_strides = a.shape() != output.shape() || b.shape() != output.shape();
  size_t output_size = output.size();

  // Choose vectorization based on tensor size
  // Use vectorization for larger tensors to amortize LLVM compilation cost
  int vector_width = 1;  // Default: scalar
  if (!needs_strides && output_size >= 10000) {
    // Use 4-wide vectors for medium/large tensors (SSE/AVX compatible)
    vector_width = 4;
  }

  // Create cache key
  codegen::KernelCacheKey cache_key{
      .operation = "add",
      .vector_width = vector_width,
      .tensor_size = output_size,
      .use_strides = needs_strides};

  // Try to get from cache or compile
  auto kernel = kernel_cache_->GetOrCompile(cache_key, [this, output_size, needs_strides, vector_width]() {
    std::unique_ptr<llvm::Module> module;
    if (vector_width > 1) {
      module = this->GenerateVectorizedAddKernelModule(output_size, vector_width);
    } else {
      module = this->GenerateAddKernelModule(output_size, needs_strides);
    }
    return jit_compiler_->CompileAddKernel(std::move(module));
  });

  // Execute the compiled kernel
  ExecuteCompiledKernel(kernel, a, b, output);
}

void JITExecutor::ExecuteCompiledKernel(
    const codegen::CompiledKernel& kernel,
    const Tensor& a, const Tensor& b, Tensor& output) {
  std::vector<int> output_shape = output.shape();
  std::vector<int> a_broadcast_strides =
      BroadcastResolver::compute_strides_for_broadcast(a.shape(), output_shape);
  std::vector<int> b_broadcast_strides =
      BroadcastResolver::compute_strides_for_broadcast(b.shape(), output_shape);

  size_t total_size = output.size();

  // For simple case without broadcasting, call kernel directly
  if (a.shape() == output.shape() && b.shape() == output.shape()) {
    kernel(const_cast<float*>(a.data()),
           const_cast<float*>(b.data()),
           output.data(),
           total_size);
  } else {
    // For broadcasted case, still use compiled kernel but with offset calculations
    // This is a simplified approach - in the future, we can generate specialized kernels
    // for broadcast patterns
    auto* a_data = const_cast<float*>(a.data());
    auto* b_data = const_cast<float*>(b.data());
    auto* out_data = output.data();

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

      out_data[i] = a_data[idx_a] + b_data[idx_b];
    }
  }
}

void JITExecutor::SetOptimizationLevel(int level) {
  jit_compiler_->SetOptimizationLevel(level);
}

int JITExecutor::GetOptimizationLevel() const {
  return jit_compiler_->GetOptimizationLevel();
}

JITExecutor::CacheStats JITExecutor::GetCacheStats() const {
  auto stats = kernel_cache_->GetStats();
  return CacheStats{
      .total_lookups = stats.total_lookups,
      .cache_hits = stats.cache_hits,
      .cache_misses = stats.cache_misses,
      .hit_rate = stats.HitRate()};
}

void JITExecutor::ResetCacheStats() {
  kernel_cache_->ResetStats();
}

void JITExecutor::ClearCache() {
  kernel_cache_->Clear();
}

size_t JITExecutor::CachedKernelCount() const {
  return kernel_cache_->Size();
}

}  // namespace mlc
