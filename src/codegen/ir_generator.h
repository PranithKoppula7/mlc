#pragma once

#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <memory>
#include <vector>
#include <functional>

namespace mlc::codegen {

class LLVMModuleBuilder;

/// Generates LLVM IR for tensor operations
/// Converts high-level tensor operations into low-level LLVM IR
class IRGenerator {
 public:
  explicit IRGenerator(LLVMModuleBuilder* builder);
  ~IRGenerator() = default;

  /// Generate IR for tensor addition kernel
  /// Signature: void add(float* a, float* b, float* c, int n)
  /// Implements: c[i] = a[i] + b[i] for i in [0, n)
  llvm::Function* GenerateAddKernel();

  /// Generate IR for tensor addition with strides (for broadcasting)
  /// Signature: void add_strided(float* a, float* b, float* c, int n,
  ///                             int stride_a, int stride_b, int stride_c)
  llvm::Function* GenerateAddKernelWithStrides();

  /// Generate vectorized add kernel (vector width = 4, 8, or 16)
  llvm::Function* GenerateVectorizedAddKernel(int vector_width);

 private:
  LLVMModuleBuilder* builder_;

  /// Helper to create a simple loop structure
  /// Creates: for (i = 0; i < n; i++) { loop_body(...); }
  void CreateLoopWithCallback(
      llvm::Function* func,
      llvm::Value* loop_count,
      std::function<void(llvm::IRBuilder<>&, llvm::Value*)> loop_body);

  /// Helper to create vector load operation
  llvm::Value* CreateVectorLoad(llvm::IRBuilder<>& builder,
                                  llvm::Value* ptr,
                                  int vector_width);

  /// Helper to create vector add operation
  llvm::Value* CreateVectorAdd(llvm::IRBuilder<>& builder,
                                 llvm::Value* a,
                                 llvm::Value* b,
                                 int vector_width);

  /// Helper to create vector store operation
  void CreateVectorStore(llvm::IRBuilder<>& builder,
                         llvm::Value* value,
                         llvm::Value* ptr);
};

}  // namespace mlc::codegen
