#include "llvm_context.h"

namespace mlc::codegen {

std::unique_ptr<llvm::LLVMContext> LLVMContextManager::context_;
std::mutex LLVMContextManager::context_mutex_;
bool LLVMContextManager::initialized_ = false;

llvm::LLVMContext& LLVMContextManager::GetContext() {
  std::lock_guard<std::mutex> lock(context_mutex_);
  if (!context_) {
    context_ = std::make_unique<llvm::LLVMContext>();
  }
  return *context_;
}

void LLVMContextManager::Initialize() {
  std::lock_guard<std::mutex> lock(context_mutex_);
  if (!initialized_) {
    // Initialize LLVM subsystems
    initialized_ = true;
  }
}

void LLVMContextManager::Shutdown() {
  std::lock_guard<std::mutex> lock(context_mutex_);
  context_.reset();
  initialized_ = false;
}

}  // namespace mlc::codegen
