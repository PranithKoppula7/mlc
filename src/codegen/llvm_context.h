#pragma once

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <memory>
#include <mutex>

namespace mlc::codegen {

/// Global LLVM context manager for MLC tensor compiler
/// Manages the global LLVM context and provides thread-safe access
class LLVMContextManager {
 public:
  /// Get the global LLVM context (singleton)
  static llvm::LLVMContext& GetContext();
  
  /// Initialize LLVM (call once at program startup)
  static void Initialize();
  
  /// Cleanup LLVM resources (call at program shutdown)
  static void Shutdown();

 private:
  LLVMContextManager() = delete;
  ~LLVMContextManager() = delete;

  static std::unique_ptr<llvm::LLVMContext> context_;
  static std::mutex context_mutex_;
  static bool initialized_;
};

}  // namespace mlc::codegen
