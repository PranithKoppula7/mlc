#!/usr/bin/env python3
"""
Phase 2 Performance Benchmarking Suite

Measures performance of JIT-compiled operations vs naive CPU paths
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'lib'))

from mlc import Tensor, enable_jit, disable_jit, get_use_jit
import time

def benchmark_operation(name, tensor_size, num_iterations=5):
    """Benchmark a tensor operation and compare JIT vs naive."""
    
    # Create test data
    data = [float(i) for i in range(tensor_size)]
    shape = [tensor_size]
    
    print(f"\n{name} (size={tensor_size} elements):")
    print("-" * 60)
    
    # Benchmark JIT path
    enable_jit()
    times_jit = []
    for _ in range(num_iterations):
        a = Tensor(data, shape=shape)
        b = Tensor(data, shape=shape)
        
        start = time.perf_counter()
        c = a + b
        result = c.to_vector()  # Force computation
        elapsed = time.perf_counter() - start
        times_jit.append(elapsed * 1e6)  # Convert to microseconds
    
    avg_jit = sum(times_jit) / len(times_jit)
    
    # Benchmark naive path
    disable_jit()
    times_naive = []
    for _ in range(num_iterations):
        a = Tensor(data, shape=shape)
        b = Tensor(data, shape=shape)
        
        start = time.perf_counter()
        c = a + b
        result = c.to_vector()  # Force computation
        elapsed = time.perf_counter() - start
        times_naive.append(elapsed * 1e6)  # Convert to microseconds
    
    avg_naive = sum(times_naive) / len(times_naive)
    
    # Calculate speedup
    speedup = avg_naive / avg_jit if avg_jit > 0 else 0
    improvement = ((avg_naive - avg_jit) / avg_naive) * 100 if avg_naive > 0 else 0
    
    # Print results
    print(f"  JIT Path    : {avg_jit:10.2f} µs (avg of {num_iterations} runs)")
    print(f"  Naive Path  : {avg_naive:10.2f} µs (avg of {num_iterations} runs)")
    print(f"  Speedup     : {speedup:10.2f}x")
    print(f"  Improvement : {improvement:10.1f}%")
    
    # Restore JIT state
    enable_jit()
    
    return {
        'name': name,
        'size': tensor_size,
        'jit_µs': avg_jit,
        'naive_µs': avg_naive,
        'speedup': speedup,
        'improvement': improvement
    }

def benchmark_broadcast():
    """Benchmark broadcasting performance."""
    print("\n" + "="*60)
    print("BROADCAST OPERATIONS")
    print("="*60)
    
    # (1000000,) + (1,) = (1000000,) - scalar broadcast
    data_a = [1.0] * 1000000
    data_b = [2.0]
    
    enable_jit()
    start = time.perf_counter()
    a = Tensor(data_a, shape=[1000000])
    b = Tensor(data_b, shape=[1])
    c = a + b
    result = c.to_vector()
    jit_time = (time.perf_counter() - start) * 1e6
    
    disable_jit()
    start = time.perf_counter()
    a = Tensor(data_a, shape=[1000000])
    b = Tensor(data_b, shape=[1])
    c = a + b
    result = c.to_vector()
    naive_time = (time.perf_counter() - start) * 1e6
    
    enable_jit()
    
    speedup = naive_time / jit_time if jit_time > 0 else 0
    improvement = ((naive_time - jit_time) / naive_time) * 100 if naive_time > 0 else 0
    
    print(f"\nScalar Broadcast (1000000,) + (1,):")
    print("-" * 60)
    print(f"  JIT Path    : {jit_time:10.2f} µs")
    print(f"  Naive Path  : {naive_time:10.2f} µs")
    print(f"  Speedup     : {speedup:10.2f}x")
    print(f"  Improvement : {improvement:10.1f}%")

def benchmark_multidimensional():
    """Benchmark multidimensional tensor operations."""
    print("\n" + "="*60)
    print("MULTIDIMENSIONAL OPERATIONS")
    print("="*60)
    
    # 2D: (1000, 1000)
    size = 1000 * 1000
    data = [1.0] * size
    
    enable_jit()
    start = time.perf_counter()
    a = Tensor(data, shape=[1000, 1000])
    b = Tensor(data, shape=[1000, 1000])
    c = a + b
    result = c.to_vector()
    jit_time = (time.perf_counter() - start) * 1e6
    
    disable_jit()
    start = time.perf_counter()
    a = Tensor(data, shape=[1000, 1000])
    b = Tensor(data, shape=[1000, 1000])
    c = a + b
    result = c.to_vector()
    naive_time = (time.perf_counter() - start) * 1e6
    
    enable_jit()
    
    speedup = naive_time / jit_time if jit_time > 0 else 0
    improvement = ((naive_time - jit_time) / naive_time) * 100 if naive_time > 0 else 0
    
    print(f"\n2D Array (1000 x 1000):")
    print("-" * 60)
    print(f"  JIT Path    : {jit_time:10.2f} µs")
    print(f"  Naive Path  : {naive_time:10.2f} µs")
    print(f"  Speedup     : {speedup:10.2f}x")
    print(f"  Improvement : {improvement:10.1f}%")

def main():
    print("\n" + "="*60)
    print("MLC TENSOR COMPILER - PHASE 2 PERFORMANCE BENCHMARKS")
    print("="*60)
    print("Comparing JIT-compiled vs naive CPU execution")
    print("="*60)
    
    results = []
    
    # Small tensor
    results.append(benchmark_operation("Small Tensor", 100, num_iterations=10))
    
    # Medium tensor
    results.append(benchmark_operation("Medium Tensor", 10000, num_iterations=5))
    
    # Large tensor
    results.append(benchmark_operation("Large Tensor", 1000000, num_iterations=3))
    
    # Broadcast operations
    benchmark_broadcast()
    
    # Multidimensional operations
    benchmark_multidimensional()
    
    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    total_speedup = 0
    for result in results:
        total_speedup += result['speedup']
    
    avg_speedup = total_speedup / len(results) if results else 1.0
    
    print(f"\nAverage Speedup: {avg_speedup:.2f}x")
    print(f"Total Operations Tested: {len(results)}")
    
    if avg_speedup > 1.0:
        print(f"\n✅ JIT compilation provides {(avg_speedup-1)*100:.1f}% performance improvement on average")
    else:
        print("\n⚠️  Note: JIT overhead may dominate for small operations")
        print("   JIT benefits increase with tensor size and workload complexity")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
