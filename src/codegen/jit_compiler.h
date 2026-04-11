#pragma once

#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <functional>
#include <memory>

namespace mlc::codegen {

/// Compiled tensor operation kernel
using CompiledKernel = std::function<void(float*, float*, float*, int)>;

/// JIT compiler for LLVM IR modules
/// Uses LLVM's OrcJIT for modern, optimized compilation
class JITCompiler {
 public:
  JITCompiler();
  ~JITCompiler() = default;

  /// Compile an LLVM module to a callable kernel
  /// Returns a function that can be called multiple times
  CompiledKernel CompileAddKernel(std::unique_ptr<llvm::Module> module);

  /// Set optimization level (0-3, default: 2)
  void SetOptimizationLevel(int level);

  /// Get current optimization level
  int GetOptimizationLevel() const { return opt_level_; }

 private:
  std::unique_ptr<llvm::orc::LLJIT> jit_;
  int opt_level_;

  /// Initialize JIT with default settings
  void InitializeJIT();
};

}  // namespace mlc::codegen
