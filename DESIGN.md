# Tensor Compiler Design: CUDA-Accelerated Tensor Addition

## 1. Overall Architecture

### High-Level Flow
```
Python API Layer (pybind11)
        ↓
Tensor Graph Builder (C++)
        ↓
Operation Scheduler (identifies parallelism)
        ↓
Code Generation Layer (LLVM IR + CUDA PTX)
        ↓
Runtime Execution (GPU/CPU hybrid)
```

### Core Philosophy
- **Simplicity First**: Focus only on tensor addition as the proof-of-concept
- **Lazy Evaluation**: Build computation graphs, optimize before execution
- **Transparent GPU**: Hide CUDA complexity behind C++ abstractions
- **Minimal Dependencies**: Leverage LLVM for code generation, pybind11 for Python binding

---

## 2. Detailed Design

### 2.1 Tensor Class

#### Current State
- Holds data as `buffer<float>` and `vector<float>`
- Tracks computation graph via `parents_` vector
- Supports operator overloading (`+`)

#### Enhancements Needed
```cpp
class Tensor {
  // Core Data
  buffer<float> data_;           // GPU/CPU agnostic allocation
  std::vector<int> shape_;       // Multi-dimensional support
  std::vector<int> strides_;     // For efficient broadcasting
  
  // Metadata
  bool requires_grad_;           // For autograd
  Device device_;                // GPU / CPU
  DataType dtype_;               // float32, float64, etc.
  
  // Graph Information
  Operation* op_;                // Which operation created this tensor
  std::vector<Tensor*> inputs_;  // Source tensors
  
  // Methods
  Tensor add(const Tensor& other);
  void to_device(Device d);
  void flatten();
  void reshape(std::vector<int> new_shape);
};
```

**Key Features**:
- `device_` field enables CPU/GPU abstraction without explicit kernel switching
- `strides_` enables efficient memory traversal for broadcasting
- `op_` and `inputs_` form the computation graph directly (vs. separate parent tracking)

---

### 2.2 Operations

#### Operation Base Class
```cpp
class Operation {
  virtual std::vector<Tensor> forward(const std::vector<Tensor>& inputs) = 0;
  virtual std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) = 0;
  std::string name();
  std::vector<int> compute_output_shape(std::vector<Tensor>& inputs);
};
```

#### Addition Operation
```cpp
class AddOp : public Operation {
  // forward: element-wise addition with broadcasting
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
    // 1. Check input compatibility (rank <= 4, shapes broadcastable)
    // 2. Compute output shape via broadcasting rules
    // 3. Select kernel based on:
    //    - Tensor layout (contiguous, strided, transposed)
    //    - Device (GPU/CPU)
    //    - Tensor sizes
    // 4. Execute kernel
    // 5. Return result
  }
  
  std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override {
    // For addition: grad_A = grad_out, grad_B = grad_out
    // Handle broadcasting reduction
  }
};
```

#### Broadcasting Support
```cpp
class BroadcastResolver {
  // Computes output shape from two inputs
  static std::vector<int> compute_output_shape(
    const std::vector<int>& shape_a,
    const std::vector<int>& shape_b
  );
  
  // Generates strides for aligned memory access
  static std::vector<int> compute_strides_for_broadcast(
    const std::vector<int>& original_shape,
    const std::vector<int>& broadcast_shape
  );
};
```

---

### 2.3 LLVM CUDA Code Generation

#### Architecture
```
Computation Graph → Optimization Pass → LLVM IR Generator
                                             ↓
                                    LLVM IR Module
                                             ↓
                        CUDA Code Generator (PTX) / CPU (x86)
                                             ↓
                                        Compiled Kernel
```

#### LLVM IR Generation Strategy

**Phase 1: IR for CPU Backend**
```cpp
class IRGenerator {
  llvm::Module* module_;
  llvm::IRBuilder<> builder_;
  
  // Generate LLVM IR for tensor operations
  llvm::Function* generate_add_kernel_cpu(
    int size,
    llvm::Type* element_type
  ) {
    // Create function: void add(float* a, float* b, float* c, int n)
    // Generate loop unrolling for vectorization:
    //   for (i = 0; i < n; i += 4) {
    //     c[i:i+4] = a[i:i+4] + b[i:i+4]  // 4-wide SIMD
    //   }
    // LLVM auto-vectorizes to AVX/SSE instructions
  }
};
```

**Phase 2: CUDA PTX Generation**
```cpp
class CUDACodeGenerator {
  // Strategy: Generate CUDA C, compile with nvcc
  // Simpler than direct PTX generation
  
  std::string generate_add_kernel(const Tensor& a, const Tensor& b) {
    return R"CUDA(
      __global__ void tensor_add_kernel(
        float* a, float* b, float* c,
        int size, int* strides_a, int* strides_b
      ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
          // Compute actual element indices based on strides
          int elem_a = compute_index(idx, strides_a);
          int elem_b = compute_index(idx, strides_b);
          c[idx] = a[elem_a] + b[elem_b];
        }
      }
    )CUDA";
  }
};
```

#### Optimization Passes
```cpp
class OptimizationPipeline {
  // 1. Fusion: Fuse consecutive operations (e.g., add → relu)
  // 2. Tiling: Optimize memory access patterns
  // 3. Loop Unrolling: For vectorization
  // 4. Register Allocation: Minimize memory bandwidth
  
  llvm::Module* optimize(llvm::Module* module) {
    llvm::PassManager pm;
    pm.add(llvm::createBasicBlockVectorizePass());
    pm.add(llvm::createLoopUnrollPass());
    pm.run(*module);
    return module;
  }
};
```

#### Kernel Compilation
```cpp
class KernelCompiler {
  // For CPU: Use LLVM JIT compiler
  JITCompiledKernel compile_cpu(llvm::Module* ir_module);
  
  // For CUDA: Delegate to nvcc or use CUDA runtime API
  // Option A: Shell out to nvcc (simple, reliable)
  // Option B: Use CUDA runtime compilation API (complex, faster)
  CUDAKernel compile_cuda(const std::string& cuda_source);
};
```

---

### 2.4 Execution Model

#### Dispatch Logic
```cpp
class KernelDispatcher {
  Tensor execute_add(const Tensor& a, const Tensor& b) {
    // Decision tree:
    if (device == GPU && size > LARGE_THRESHOLD) {
      return execute_on_gpu(a, b);
    } else if (device == CPU && size > SMALL_THRESHOLD) {
      return execute_vectorized_cpu(a, b);
    } else {
      return execute_naive_cpu(a, b);
    }
  }
  
private:
  Tensor execute_on_gpu(const Tensor& a, const Tensor& b);
  Tensor execute_vectorized_cpu(const Tensor& a, const Tensor& b);
  Tensor execute_naive_cpu(const Tensor& a, const Tensor& b);
};
```

#### Memory Management
```cpp
class MemoryAllocator {
  // GPU: Use CUDA memory pool for efficient allocation
  // CPU: Use aligned malloc for SIMD efficiency
  
  float* allocate_device(size_t bytes, Device device);
  void deallocate_device(float* ptr, Device device);
  
  // Automatic caching of compiled kernels
  std::unordered_map<std::string, CompiledKernel> kernel_cache_;
};
```

---

## 3. Python Library

### API Surface
```python
import fsml

# Create tensors
a = fsml.Tensor([1.0, 2.0, 3.0], shape=(3,))
b = fsml.Tensor([4.0, 5.0, 6.0], shape=(3,))

# Add operation (lazy evaluation)
c = a + b

# Explicit computation
result = c.compute()  # or c.numpy()

# Device placement
a_gpu = a.to_device("cuda:0")
b_gpu = b.to_device("cuda:0")
c_gpu = a_gpu + b_gpu
print(c_gpu.numpy())  # Copy back to CPU for inspection

# Broadcasting
x = fsml.Tensor([[1.0, 2.0]], shape=(1, 2))
y = fsml.Tensor([10.0], shape=(1,))
z = x + y  # Broadcasting automatically applied
```

### Implementation Strategy
```cpp
// In python/tensor.cpp
PYBIND11_MODULE(core, m) {
  py::class_<Tensor>(m, "Tensor")
    .def(py::init<std::vector<float>, std::vector<int>>())
    .def("__add__", &Tensor::add)
    .def("compute", &Tensor::compute)
    .def("numpy", &Tensor::to_numpy)
    .def("to_device", &Tensor::to_device)
    .def("shape", &Tensor::shape)
    .def("__repr__", &Tensor::repr);
}
```

### Additional Python Utilities
```python
# fsml/__init__.py
from .core import Tensor

def zeros(shape, device="cpu"):
    return Tensor([0.0] * np.prod(shape), shape)

def ones(shape, device="cpu"):
    return Tensor([1.0] * np.prod(shape), shape)

def random(shape, device="cpu"):
    return Tensor(np.random.randn(*shape).tolist(), shape)
```

---

## 4. Implementation Roadmap

### Phase 1: CPU Foundation (Week 1)
- [ ] Enhance Tensor class with metadata
- [ ] Implement AddOp with shape validation
- [ ] Add BroadcastResolver
- [ ] Create simple CPU execution path
- [ ] Python bindings for basic operations

### Phase 2: Code Generation (Week 2)
- [ ] LLVM IR generation for simple addition
- [ ] CPU JIT compilation and execution
- [ ] Loop unrolling pass for vectorization
- [ ] Performance profiling infrastructure

### Phase 3: GPU Support (Week 3)
- [ ] CUDA kernel wrapper (simple inline CUDA strings)
- [ ] GPU memory management
- [ ] Device dispatch logic
- [ ] GPU JIT compilation

### Phase 4: Optimization & Polish (Week 4)
- [ ] Kernel caching
- [ ] Graph optimization passes (fusion, etc.)
- [ ] Benchmarking suite
- [ ] Documentation and examples

---

## 5. Tradeoffs & Miscellaneous

### Design Tradeoffs

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| IR Backend | LLVM | Mature, auto-vectorization, wide platform support |
| CUDA Strategy | Delegate to nvcc | Simpler than PTX generation, leverages NVidia's toolchain |
| Code Generation | C++ strings → nvcc | Fast iteration, easier to debug than raw PTX |
| Kernel Caching | In-memory cache | Avoids recompilation, handles typical workloads |
| Broadcasting | Lazy computation | Only compute what's needed, efficient memory |
| Memory Model | Explicit device placement | Matches modern frameworks (PyTorch), clearer API |

### Optimization Opportunities (Future)

1. **Fused Operations**: Add-ReLU, Add-Normalize kernel fusion
2. **Tiling**: Optimize cache locality for large tensors
3. **Half Precision**: fp16 kernels for 2x throughput
4. **Distributed**: Multi-GPU support via NCCL
5. **Auto-tuning**: Profiling-driven kernel specialization

### Performance Targets

**Single Addition Benchmark** (target for 1M elements):
- CPU (vectorized): ~100-200µs
- GPU (simple kernel): ~500µs (setup dominates)
- GPU (batched): ~50µs per op (amortized)

### Potential Pitfalls & Mitigations

| Pitfall | Mitigation |
|---------|-----------|
| CUDA kernel divergence | Use warp-aware primitives, handle divergence explicitly |
| Broadcasting complexity | Precompute strides, test extensively |
| Memory fragmentation (GPU) | Use memory pool allocator from the start |
| Compilation latency | Implement aggressive kernel caching |
| Python GIL blocking | Release GIL during GPU launches |

### Minimal Example: End-to-End Flow

**Input**:
```python
a = fsml.Tensor([1, 2, 3], shape=(3,))
b = fsml.Tensor([4, 5, 6], shape=(3,))
c = a + b
print(c.compute())  # [5, 7, 9]
```

**Execution Flow**:
1. Python calls `__add__` → C++ `Tensor::operator+()`
2. Creates `AddOp` and builds compute graph
3. Python calls `compute()`
4. Graph executor traverses: A, B → AddOp
5. BroadcastResolver checks shapes (both (3,), no broadcast needed)
6. Dispatcher selects CPU vectorized kernel (small tensor)
7. LLVM IR generator creates optimized `add_kernel_cpu`
8. JIT compiler emits x86 SIMD instructions
9. Kernel executes: `c[i] = a[i] + b[i]` with AVX2 vectorization
10. Result copied back to Python as numpy array

---

## Summary

This design prioritizes **simplicity and correctness** while building a foundation for GPU acceleration. By starting with CPU LLVM IR generation, we validate the compilation pipeline before adding CUDA complexity. The lazy evaluation model enables future optimizations like operation fusion without changing the API.

**Key Insight**: The compiler transforms tensor operations into specialized, vectorized machine code. By leveraging LLVM and CUDA toolchains, we avoid reimplementing optimization passes and hardware-specific code generation.
