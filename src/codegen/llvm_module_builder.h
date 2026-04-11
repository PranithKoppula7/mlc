#pragma once

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <memory>
#include <string>

namespace mlc::codegen {

/// Builder for creating LLVM IR modules and functions
/// Provides convenient interface for IR generation
class LLVMModuleBuilder {
 public:
  explicit LLVMModuleBuilder(const std::string& module_name);
  ~LLVMModuleBuilder() = default;

  /// Get the underlying LLVM module
  llvm::Module* GetModule() { return module_.get(); }

  /// Get the IR builder for code generation
  llvm::IRBuilder<>& GetBuilder() { return builder_; }

  /// Get the LLVM context
  llvm::LLVMContext& GetContext() { return context_; }

  /// Create a new function with given name, return type, and argument types
  llvm::Function* CreateFunction(
      const std::string& name,
      llvm::Type* return_type,
      const std::vector<llvm::Type*>& arg_types);

  /// Create a basic block and position builder at the end
  llvm::BasicBlock* CreateBasicBlock(llvm::Function* func,
                                      const std::string& name = "");

  /// Verify the module and print errors if any
  bool VerifyModule(std::string* error_msg = nullptr);

  /// Dump the module to string (for debugging)
  std::string DumpModule();

 private:
  llvm::LLVMContext& context_;
  std::unique_ptr<llvm::Module> module_;
  llvm::IRBuilder<> builder_;
};

}  // namespace mlc::codegen
