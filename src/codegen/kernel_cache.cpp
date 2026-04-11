#include "kernel_cache.h"
#include <algorithm>

namespace mlc::codegen {

KernelCache::KernelCache()
    : total_lookups_(0), cache_hits_(0), cache_misses_(0), 
      max_cache_size_(100) {}  // Default max 100 kernels

CompiledKernel KernelCache::GetOrCompile(
    const KernelCacheKey& key,
    std::function<CompiledKernel()> generator) {
  // Update statistics
  {
    std::lock_guard<std::mutex> stats_lock(stats_mutex_);
    total_lookups_++;
  }

  // Check cache
  {
    std::lock_guard<std::mutex> cache_lock(cache_mutex_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      std::lock_guard<std::mutex> stats_lock(stats_mutex_);
      cache_hits_++;
      
      // Update LRU: move to end (most recently used)
      auto pos = std::find(access_order_.begin(), access_order_.end(), key);
      if (pos != access_order_.end()) {
        access_order_.erase(pos);
      }
      access_order_.push_back(key);
      
      return it->second;
    }
  }

  // Cache miss - compile kernel
  CompiledKernel kernel = generator();

  // Store in cache
  {
    std::lock_guard<std::mutex> cache_lock(cache_mutex_);
    
    // Check if we need to evict
    if (cache_.size() >= max_cache_size_) {
      // Remove LRU entry (least recently used)
      if (!access_order_.empty()) {
        const auto& lru_key = access_order_.front();
        access_order_.erase(access_order_.begin());
        cache_.erase(lru_key);
      }
    }
    
    cache_[key] = kernel;
    access_order_.push_back(key);
    
    std::lock_guard<std::mutex> stats_lock(stats_mutex_);
    cache_misses_++;
  }

  return kernel;
}

void KernelCache::Clear() {
  std::lock_guard<std::mutex> cache_lock(cache_mutex_);
  cache_.clear();
}

size_t KernelCache::Size() const {
  std::lock_guard<std::mutex> cache_lock(cache_mutex_);
  return cache_.size();
}

KernelCache::Stats KernelCache::GetStats() const {
  std::lock_guard<std::mutex> stats_lock(stats_mutex_);
  return Stats{total_lookups_, cache_hits_, cache_misses_};
}

void KernelCache::ResetStats() {
  std::lock_guard<std::mutex> stats_lock(stats_mutex_);
  total_lookups_ = 0;
  cache_hits_ = 0;
  cache_misses_ = 0;
}

void KernelCache::SetMaxCacheSize(size_t max_size) {
  std::lock_guard<std::mutex> cache_lock(cache_mutex_);
  max_cache_size_ = max_size;
  
  // Evict oldest entries if necessary
  while (cache_.size() > max_cache_size_ && !access_order_.empty()) {
    const auto& lru_key = access_order_.front();
    access_order_.erase(access_order_.begin());
    cache_.erase(lru_key);
  }
}

size_t KernelCache::GetMaxCacheSize() const {
  std::lock_guard<std::mutex> cache_lock(cache_mutex_);
  return max_cache_size_;
}

void KernelCache::PrecompileKernel(const KernelCacheKey& key,
                                   std::function<CompiledKernel()> generator) {
  GetOrCompile(key, generator);
}

}  // namespace mlc::codegen
