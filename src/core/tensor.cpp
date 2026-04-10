#include "tensor.h"
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <sstream>
#include <iomanip>

namespace mlc {

Tensor::Tensor()
    : requires_grad_(false),
      device_(Device::CPU), dtype_(DataType::FLOAT32),
      op_(nullptr) {}

Tensor::Tensor(const std::vector<float>& data, const std::vector<int>& shape)
    : requires_grad_(false),
      device_(Device::CPU), dtype_(DataType::FLOAT32),
      shape_(shape), op_(nullptr) {
    size_t total_size = std::accumulate(shape.begin(), shape.end(), 
                                        1, std::multiplies<int>());
    if (data.size() != total_size) {
        throw std::invalid_argument("Data size does not match shape");
    }
    allocate(total_size);
    std::copy(data.begin(), data.end(), data_.get());
    compute_strides();
}

Tensor::Tensor(const std::vector<int>& shape, Device device)
    : requires_grad_(false),
      device_(device), dtype_(DataType::FLOAT32),
      shape_(shape), op_(nullptr) {
    size_t total_size = std::accumulate(shape.begin(), shape.end(), 
                                        1, std::multiplies<int>());
    allocate(total_size);
    std::memset(data_.get(), 0, total_size * sizeof(float));
    compute_strides();
}

Tensor::Tensor(const Tensor& other)
    : requires_grad_(other.requires_grad_),
      device_(other.device_), dtype_(other.dtype_),
      shape_(other.shape_), strides_(other.strides_),
      op_(other.op_) {
    size_t total_size = other.size();
    if (total_size > 0) {
        allocate(total_size);
        std::copy(other.data_.get(), other.data_.get() + total_size, data_.get());
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        requires_grad_ = other.requires_grad_;
        device_ = other.device_;
        dtype_ = other.dtype_;
        shape_ = other.shape_;
        strides_ = other.strides_;
        op_ = other.op_;
        inputs_ = other.inputs_;
        
        size_t total_size = other.size();
        if (total_size > 0) {
            allocate(total_size);
            std::copy(other.data_.get(), other.data_.get() + total_size, data_.get());
        } else {
            data_.reset();
        }
    }
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_(std::move(other.data_)),
      shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)),
      requires_grad_(other.requires_grad_),
      device_(other.device_),
      dtype_(other.dtype_),
      op_(other.op_),
      inputs_(std::move(other.inputs_)) {}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        data_ = std::move(other.data_);
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        requires_grad_ = other.requires_grad_;
        device_ = other.device_;
        dtype_ = other.dtype_;
        op_ = other.op_;
        inputs_ = std::move(other.inputs_);
    }
    return *this;
}

Tensor::~Tensor() = default;

void Tensor::allocate(size_t size) {
    if (size > 0) {
        data_ = std::make_unique<float[]>(size);
    } else {
        data_.reset();
    }
}

void Tensor::compute_strides() {
    strides_.clear();
    if (shape_.empty()) {
        return;
    }
    
    strides_.resize(shape_.size(), 1);
    for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

size_t Tensor::size() const {
    return std::accumulate(shape_.begin(), shape_.end(), 
                          1, std::multiplies<int>());
}

Tensor Tensor::add(const Tensor& other) {
    // The actual add operation will be done through the Operation interface
    // For Phase 1, this is a placeholder that builds the computation graph
    throw std::runtime_error("add() should be called through AddOp::forward()");
}

Tensor Tensor::to_device(Device device) {
    if (device == device_) {
        return *this;
    }
    
    Tensor result(shape_, device);
    if (size() > 0) {
        std::copy(data_.get(), data_.get() + size(), result.data_.get());
    }
    return result;
}

Tensor Tensor::flatten() {
    size_t total_size = size();
    std::vector<int> new_shape = {static_cast<int>(total_size)};
    return reshape(new_shape);
}

Tensor Tensor::reshape(const std::vector<int>& new_shape) {
    size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 
                                      1, std::multiplies<int>());
    if (new_size != size()) {
        throw std::invalid_argument("Cannot reshape: total elements mismatch");
    }
    
    Tensor result(new_shape, device_);
    if (size() > 0) {
        std::copy(data_.get(), data_.get() + size(), result.data_.get());
    }
    return result;
}

void Tensor::set_op(Operation* op, const std::vector<Tensor*>& inputs) {
    op_ = op;
    inputs_ = inputs;
}

std::string Tensor::repr() const {
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape_[i];
    }
    oss << "], device=" << (device_ == Device::CPU ? "CPU" : "CUDA");
    oss << ", dtype=FLOAT32, data=[";
    
    size_t total_size = size();
    for (size_t i = 0; i < std::min(size_t(5), total_size); ++i) {
        if (i > 0) oss << ", ";
        oss << std::fixed << std::setprecision(4) << data_[i];
    }
    if (total_size > 5) {
        oss << ", ... (" << total_size << " elements)";
    }
    oss << "])";
    return oss.str();
}

std::vector<float> Tensor::to_vector() const {
    size_t total_size = size();
    if (total_size == 0) {
        return {};
    }
    return std::vector<float>(data_.get(), data_.get() + total_size);
}

}  // namespace mlc
