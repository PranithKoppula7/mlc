#include "jit_compiler.h"
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/TargetSelect.h>
#include <stdexcept>

namespace mlc::codegen {

JITCompiler::JITCompiler() : opt_level_(2) {
  InitializeJIT();
}

void JITCompiler::InitializeJIT() {
  // Initialize LLVM for native target
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  // Create LLJIT
  auto jit_result = llvm::orc::LLJITBuilder().create();
  if (!jit_result) {
    throw std::runtime_error("Failed to create LLJIT");
  }
  jit_ = std::move(jit_result.get());
}

CompiledKernel JITCompiler::CompileAddKernel(
    std::unique_ptr<llvm::Module> module) {
  if (!jit_) {
    throw std::runtime_error("JIT not initialized");
  }

  // Apply optimization passes
  llvm::PassBuilder pb;
  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  pb.registerLoopAnalyses(lam);
  pb.registerFunctionAnalyses(fam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerModuleAnalyses(mam);

  pb.crossRegisterProxies(lam, fam, cgam, mam);

  llvm::ModulePassManager mpm;
  if (opt_level_ >= 1) {
    llvm::OptimizationLevel level;
    switch (opt_level_) {
      case 1:
        level = llvm::OptimizationLevel::O1;
        break;
      case 2:
        level = llvm::OptimizationLevel::O2;
        break;
      case 3:
        level = llvm::OptimizationLevel::O3;
        break;
      default:
        level = llvm::OptimizationLevel::O0;
    }
    
    // Build the default optimization pipeline
    mpm.addPass(pb.buildPerModuleDefaultPipeline(level));
    
    // Add additional passes for tensor operations (O2+)
    if (opt_level_ >= 2) {
      // The default pipeline already includes loop vectorization and unrolling
      // We rely on LLVM's auto-vectorization for SSE/AVX instructions
    }
  }
  mpm.run(*module, mam);

  // Add module to JIT
  auto tsm = llvm::orc::ThreadSafeModule(std::move(module),
                                          std::make_unique<llvm::LLVMContext>());
  if (auto err = jit_->addIRModule(std::move(tsm))) {
    throw std::runtime_error("Failed to add module to JIT");
  }

  // Lookup "add" function
  auto add_sym = jit_->lookup("add");
  if (!add_sym) {
    throw std::runtime_error("Failed to lookup 'add' function");
  }

  // Convert to callable function
  auto* add_fn = (void (*)(float*, float*, float*, int))(
      add_sym->getAddress());
  
  return [add_fn](float* a, float* b, float* c, int n) {
    add_fn(a, b, c, n);
  };
}

void JITCompiler::SetOptimizationLevel(int level) {
  if (level < 0 || level > 3) {
    throw std::invalid_argument("Optimization level must be 0-3");
  }
  opt_level_ = level;
}

}  // namespace mlc::codegen
