#pragma once

#include "jit_compiler.h"
#include <functional>
#include <map>
#include <mutex>
#include <string>

namespace mlc::codegen {

/// Cache key for compiled kernels
struct KernelCacheKey {
  std::string operation;      // e.g., "add", "mul"
  int vector_width;           // 1, 4, 8, 16
  size_t tensor_size;         // Optional: for size-specific specialization
  bool use_strides;           // Whether operation uses strides

  bool operator<(const KernelCacheKey& other) const {
    if (operation != other.operation) return operation < other.operation;
    if (vector_width != other.vector_width) return vector_width < other.vector_width;
    if (tensor_size != other.tensor_size) return tensor_size < other.tensor_size;
    return use_strides < other.use_strides;
  }

  bool operator==(const KernelCacheKey& other) const {
    return operation == other.operation &&
           vector_width == other.vector_width &&
           tensor_size == other.tensor_size &&
           use_strides == other.use_strides;
  }
};

/// Thread-safe cache for compiled kernel functions
class KernelCache {
 public:
  KernelCache();
  ~KernelCache() = default;

  /// Get or compile a kernel
  /// If kernel is in cache, returns immediately
  /// Otherwise, calls generator, compiles, caches, and returns
  CompiledKernel GetOrCompile(
      const KernelCacheKey& key,
      std::function<CompiledKernel()> generator);

  /// Clear all cached kernels
  void Clear();

  /// Get current cache size (number of cached kernels)
  size_t Size() const;

  /// Get cache hit rate statistics
  struct Stats {
    size_t total_lookups;
    size_t cache_hits;
    size_t cache_misses;
    
    double HitRate() const {
      return total_lookups == 0 ? 0.0 : 
             static_cast<double>(cache_hits) / total_lookups;
    }
  };

  /// Get cache statistics
  Stats GetStats() const;

  /// Reset statistics
  void ResetStats();
  
  /// Set maximum cache size (number of kernels)
  /// When exceeded, uses LRU eviction
  void SetMaxCacheSize(size_t max_size);
  
  /// Get maximum cache size
  size_t GetMaxCacheSize() const;
  
  /// Precompile and cache a kernel with given parameters
  /// Useful for warming up cache with common operations
  void PrecompileKernel(const KernelCacheKey& key,
                        std::function<CompiledKernel()> generator);

 private:
  std::map<KernelCacheKey, CompiledKernel> cache_;
  mutable std::mutex cache_mutex_;
  
  // Statistics
  size_t total_lookups_;
  size_t cache_hits_;
  size_t cache_misses_;
  mutable std::mutex stats_mutex_;
  
  // Cache management
  size_t max_cache_size_;
  std::vector<KernelCacheKey> access_order_;  // For LRU eviction
};

}  // namespace mlc::codegen
