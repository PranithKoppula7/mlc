#include "ir_generator.h"
#include "llvm_module_builder.h"
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Type.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

namespace mlc::codegen {

IRGenerator::IRGenerator(LLVMModuleBuilder* builder) : builder_(builder) {}

llvm::Function* IRGenerator::GenerateAddKernel() {
  // Get types
  llvm::Type* float_type = llvm::Type::getFloatTy(builder_->GetContext());
  llvm::Type* ptr_float_type = float_type->getPointerTo();
  llvm::Type* int_type = llvm::Type::getInt32Ty(builder_->GetContext());
  llvm::Type* void_type = llvm::Type::getVoidTy(builder_->GetContext());

  // Create function: void add(float* a, float* b, float* c, int n)
  std::vector<llvm::Type*> arg_types = {ptr_float_type, ptr_float_type,
                                         ptr_float_type, int_type};
  llvm::Function* add_func =
      builder_->CreateFunction("add", void_type, arg_types);

  // Set argument names
  auto args = add_func->args().begin();
  llvm::Value* ptr_a = &(*args);
  ptr_a->setName("a");
  llvm::Value* ptr_b = &(*std::next(args));
  ptr_b->setName("b");
  llvm::Value* ptr_c = &(*std::next(args, 2));
  ptr_c->setName("c");
  llvm::Value* count = &(*std::next(args, 3));
  count->setName("n");

  // Create entry block
  builder_->CreateBasicBlock(add_func, "entry");
  auto& builder = builder_->GetBuilder();

  // Create loop: for (i = 0; i < n; i++)
  llvm::Value* zero = llvm::ConstantInt::get(int_type, 0);
  llvm::Value* one = llvm::ConstantInt::get(int_type, 1);

  // Loop pre-condition
  llvm::BasicBlock* loop_cond =
      llvm::BasicBlock::Create(builder_->GetContext(), "loop.cond", add_func);
  llvm::BasicBlock* loop_body =
      llvm::BasicBlock::Create(builder_->GetContext(), "loop.body", add_func);
  llvm::BasicBlock* loop_inc =
      llvm::BasicBlock::Create(builder_->GetContext(), "loop.inc", add_func);
  llvm::BasicBlock* loop_end =
      llvm::BasicBlock::Create(builder_->GetContext(), "loop.end", add_func);

  // Jump to loop condition
  builder.CreateBr(loop_cond);

  // Loop condition: if (i < n) goto loop_body; else goto loop_end;
  builder.SetInsertPoint(loop_cond);
  llvm::PHINode* i_phi = builder.CreatePHI(int_type, 2, "i");
  i_phi->addIncoming(zero, &add_func->getEntryBlock());
  llvm::Value* cond = builder.CreateICmpSLT(i_phi, count, "cond");
  builder.CreateCondBr(cond, loop_body, loop_end);

  // Loop body: c[i] = a[i] + b[i];
  builder.SetInsertPoint(loop_body);
  llvm::Value* gep_a = builder.CreateGEP(float_type, ptr_a, i_phi, "gep_a");
  llvm::Value* gep_b = builder.CreateGEP(float_type, ptr_b, i_phi, "gep_b");
  llvm::Value* gep_c = builder.CreateGEP(float_type, ptr_c, i_phi, "gep_c");
  llvm::Value* val_a = builder.CreateLoad(float_type, gep_a, "val_a");
  llvm::Value* val_b = builder.CreateLoad(float_type, gep_b, "val_b");
  llvm::Value* result = builder.CreateFAdd(val_a, val_b, "add");
  builder.CreateStore(result, gep_c);
  builder.CreateBr(loop_inc);

  // Loop increment: i++
  builder.SetInsertPoint(loop_inc);
  llvm::Value* i_next = builder.CreateAdd(i_phi, one, "i.next");
  i_phi->addIncoming(i_next, loop_inc);
  builder.CreateBr(loop_cond);

  // Loop end: return;
  builder.SetInsertPoint(loop_end);
  builder.CreateRetVoid();

  return add_func;
}

llvm::Function* IRGenerator::GenerateAddKernelWithStrides() {
  // For now, same as non-strided version
  // TODO: Implement strided access for broadcasting
  return GenerateAddKernel();
}

llvm::Function* IRGenerator::GenerateVectorizedAddKernel(int vector_width) {
  // For now, implement loop unrolling instead of true SIMD vectors
  // This is simpler and achieves similar benefits through instruction-level parallelism
  // True SIMD support will be added in Phase 3
  
  // Get types
  llvm::Type* float_type = llvm::Type::getFloatTy(builder_->GetContext());
  llvm::Type* ptr_float_type = float_type->getPointerTo();
  llvm::Type* int_type = llvm::Type::getInt32Ty(builder_->GetContext());
  llvm::Type* void_type = llvm::Type::getVoidTy(builder_->GetContext());

  // Create function: void add_vec(float* a, float* b, float* c, int n)
  std::vector<llvm::Type*> arg_types = {ptr_float_type, ptr_float_type,
                                         ptr_float_type, int_type};
  llvm::Function* add_func =
      builder_->CreateFunction("add_vec", void_type, arg_types);

  // Set argument names
  auto args = add_func->args().begin();
  llvm::Value* ptr_a = &(*args);
  ptr_a->setName("a");
  llvm::Value* ptr_b = &(*std::next(args));
  ptr_b->setName("b");
  llvm::Value* ptr_c = &(*std::next(args, 2));
  ptr_c->setName("c");
  llvm::Value* count = &(*std::next(args, 3));
  count->setName("n");

  // Create basic blocks
  builder_->CreateBasicBlock(add_func, "entry");
  auto& builder = builder_->GetBuilder();

  // Unrolled loop: for (i = 0; i < n; i += vector_width)
  // Each iteration processes vector_width elements
  llvm::Value* zero = llvm::ConstantInt::get(int_type, 0);
  llvm::Value* vec_step = llvm::ConstantInt::get(int_type, vector_width);
  llvm::Value* one = llvm::ConstantInt::get(int_type, 1);

  // Loop pre-condition
  llvm::BasicBlock* loop_cond =
      llvm::BasicBlock::Create(builder_->GetContext(), "loop.cond", add_func);
  llvm::BasicBlock* loop_body =
      llvm::BasicBlock::Create(builder_->GetContext(), "loop.body", add_func);
  llvm::BasicBlock* loop_inc =
      llvm::BasicBlock::Create(builder_->GetContext(), "loop.inc", add_func);
  llvm::BasicBlock* loop_end =
      llvm::BasicBlock::Create(builder_->GetContext(), "loop.end", add_func);

  // Jump to loop condition
  builder.CreateBr(loop_cond);

  // Loop condition: if (i + vector_width <= n)
  builder.SetInsertPoint(loop_cond);
  llvm::PHINode* i_phi = builder.CreatePHI(int_type, 2, "i");
  i_phi->addIncoming(zero, &add_func->getEntryBlock());
  
  llvm::Value* i_next_check = builder.CreateAdd(i_phi, vec_step, "i_next");
  llvm::Value* cond = builder.CreateICmpSLE(i_next_check, count, "cond");
  builder.CreateCondBr(cond, loop_body, loop_end);

  // Loop body: unrolled operations
  builder.SetInsertPoint(loop_body);
  
  // Process vector_width elements in this iteration
  for (int j = 0; j < vector_width; ++j) {
    llvm::Value* offset = llvm::ConstantInt::get(int_type, j);
    llvm::Value* idx = builder.CreateAdd(i_phi, offset, "idx_" + std::to_string(j));
    
    llvm::Value* gep_a = builder.CreateGEP(float_type, ptr_a, idx);
    llvm::Value* gep_b = builder.CreateGEP(float_type, ptr_b, idx);
    llvm::Value* gep_c = builder.CreateGEP(float_type, ptr_c, idx);
    
    llvm::Value* val_a = builder.CreateLoad(float_type, gep_a, "val_a_" + std::to_string(j));
    llvm::Value* val_b = builder.CreateLoad(float_type, gep_b, "val_b_" + std::to_string(j));
    llvm::Value* result = builder.CreateFAdd(val_a, val_b, "add_" + std::to_string(j));
    builder.CreateStore(result, gep_c);
  }
  
  builder.CreateBr(loop_inc);

  // Loop increment: i += vector_width
  builder.SetInsertPoint(loop_inc);
  llvm::Value* i_incremented = builder.CreateAdd(i_phi, vec_step, "i.next");
  i_phi->addIncoming(i_incremented, loop_inc);
  builder.CreateBr(loop_cond);

  // Scalar cleanup loop for remaining elements
  llvm::BasicBlock* scalar_loop_cond =
      llvm::BasicBlock::Create(builder_->GetContext(), "scalar.cond", add_func);
  llvm::BasicBlock* scalar_loop_body =
      llvm::BasicBlock::Create(builder_->GetContext(), "scalar.body", add_func);
  llvm::BasicBlock* scalar_loop_inc =
      llvm::BasicBlock::Create(builder_->GetContext(), "scalar.inc", add_func);

  builder.SetInsertPoint(loop_end);
  builder.CreateBr(scalar_loop_cond);

  // Scalar loop condition
  builder.SetInsertPoint(scalar_loop_cond);
  llvm::PHINode* i_scalar_phi = builder.CreatePHI(int_type, 2, "i_scalar");
  i_scalar_phi->addIncoming(i_phi, loop_end);
  
  llvm::Value* cond_scalar = builder.CreateICmpSLT(i_scalar_phi, count, "cond_scalar");
  builder.CreateCondBr(cond_scalar, scalar_loop_body, 
                      llvm::BasicBlock::Create(builder_->GetContext(), "return", add_func));

  // Scalar loop body
  builder.SetInsertPoint(scalar_loop_body);
  llvm::Value* gep_a_s = builder.CreateGEP(float_type, ptr_a, i_scalar_phi);
  llvm::Value* gep_b_s = builder.CreateGEP(float_type, ptr_b, i_scalar_phi);
  llvm::Value* gep_c_s = builder.CreateGEP(float_type, ptr_c, i_scalar_phi);
  llvm::Value* val_a_s = builder.CreateLoad(float_type, gep_a_s, "val_a_s");
  llvm::Value* val_b_s = builder.CreateLoad(float_type, gep_b_s, "val_b_s");
  llvm::Value* result_s = builder.CreateFAdd(val_a_s, val_b_s, "add_s");
  builder.CreateStore(result_s, gep_c_s);
  builder.CreateBr(scalar_loop_inc);

  // Scalar loop increment
  builder.SetInsertPoint(scalar_loop_inc);
  llvm::Value* i_scalar_next = builder.CreateAdd(i_scalar_phi, one, "i_scalar.next");
  i_scalar_phi->addIncoming(i_scalar_next, scalar_loop_inc);
  builder.CreateBr(scalar_loop_cond);

  // Final return
  auto it = add_func->end();
  --it;
  builder.SetInsertPoint(&*it);
  builder.CreateRetVoid();

  return add_func;
}

void IRGenerator::CreateLoopWithCallback(
    llvm::Function* func, llvm::Value* loop_count,
    std::function<void(llvm::IRBuilder<>&, llvm::Value*)> loop_body) {
  // Placeholder for future enhancement
}

llvm::Value* IRGenerator::CreateVectorLoad(llvm::IRBuilder<>& builder,
                                            llvm::Value* ptr,
                                            int vector_width) {
  // Placeholder for future enhancement
  return nullptr;
}

llvm::Value* IRGenerator::CreateVectorAdd(llvm::IRBuilder<>& builder,
                                           llvm::Value* a, llvm::Value* b,
                                           int vector_width) {
  // Placeholder for future enhancement
  return nullptr;
}

void IRGenerator::CreateVectorStore(llvm::IRBuilder<>& builder,
                                     llvm::Value* value,
                                     llvm::Value* ptr) {
  // Placeholder for future enhancement
}

}  // namespace mlc::codegen
