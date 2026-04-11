#include "llvm_module_builder.h"
#include "llvm_context.h"
#include <llvm/IR/Function.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <sstream>

namespace mlc::codegen {

LLVMModuleBuilder::LLVMModuleBuilder(const std::string& module_name)
    : context_(LLVMContextManager::GetContext()),
      module_(std::make_unique<llvm::Module>(module_name, context_)),
      builder_(context_) {}

llvm::Function* LLVMModuleBuilder::CreateFunction(
    const std::string& name,
    llvm::Type* return_type,
    const std::vector<llvm::Type*>& arg_types) {
  // Create function type
  llvm::FunctionType* func_type =
      llvm::FunctionType::get(return_type, arg_types, false);

  // Create function in module
  llvm::Function* func = llvm::Function::Create(
      func_type, llvm::Function::ExternalLinkage, name, module_.get());

  return func;
}

llvm::BasicBlock* LLVMModuleBuilder::CreateBasicBlock(
    llvm::Function* func, const std::string& name) {
  std::string block_name = name.empty() ? "entry" : name;
  llvm::BasicBlock* bb =
      llvm::BasicBlock::Create(context_, block_name, func);
  builder_.SetInsertPoint(bb);
  return bb;
}

bool LLVMModuleBuilder::VerifyModule(std::string* error_msg) {
  std::string error_buffer;
  llvm::raw_string_ostream error_stream(error_buffer);
  
  bool is_valid = !llvm::verifyModule(*module_, &error_stream);
  
  if (!is_valid && error_msg) {
    *error_msg = error_stream.str();
  }
  
  return is_valid;
}

std::string LLVMModuleBuilder::DumpModule() {
  std::string result;
  llvm::raw_string_ostream os(result);
  module_->print(os, nullptr);
  return result;
}

}  // namespace mlc::codegen
