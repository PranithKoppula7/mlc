#pragma once

#include <vector>
#include <memory>
#include <string>
#include <cstring>

namespace mlc {

enum class Device {
    CPU,
    CUDA
};

enum class DataType {
    FLOAT32,
    FLOAT64
};

class Operation;

class Tensor {
public:
    // Constructors
    Tensor();
    Tensor(const std::vector<float>& data, const std::vector<int>& shape);
    explicit Tensor(const std::vector<int>& shape, Device device = Device::CPU);
    
    // Copy constructor and assignment
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    
    // Move constructor and assignment
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    
    ~Tensor();
    
    // Core data access
    float* data() { return data_.get(); }
    const float* data() const { return data_.get(); }
    
    // Metadata accessors
    const std::vector<int>& shape() const { return shape_; }
    const std::vector<int>& strides() const { return strides_; }
    Device device() const { return device_; }
    DataType dtype() const { return dtype_; }
    bool requires_grad() const { return requires_grad_; }
    Operation* op() const { return op_; }
    const std::vector<Tensor*>& inputs() const { return inputs_; }
    
    // Shape utilities
    size_t size() const;
    size_t rank() const { return shape_.size(); }
    
    // Operations
    Tensor add(const Tensor& other);
    
    // Device management
    Tensor to_device(Device device);
    
    // Shape manipulation
    Tensor flatten();
    Tensor reshape(const std::vector<int>& new_shape);
    
    // Graph management
    void set_op(Operation* op, const std::vector<Tensor*>& inputs);
    
    // Utilities
    std::string repr() const;
    std::vector<float> to_vector() const;
    
private:
    // Core data
    std::unique_ptr<float[]> data_;
    
    // Metadata
    std::vector<int> shape_;
    std::vector<int> strides_;
    bool requires_grad_;
    Device device_;
    DataType dtype_;
    
    // Graph information
    Operation* op_;
    std::vector<Tensor*> inputs_;
    
    // Helper methods
    void compute_strides();
    void allocate(size_t size);
};

}  // namespace mlc
