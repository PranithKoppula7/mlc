#pragma once

#include "tensor.h"
#include <vector>
#include <string>

namespace mlc {

class Operation {
public:
    virtual ~Operation() = default;
    
    // Forward pass: compute output tensor from input tensors
    virtual std::vector<Tensor> forward(const std::vector<Tensor>& inputs) = 0;
    
    // Backward pass: compute gradients w.r.t. inputs from gradient w.r.t. outputs
    virtual std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) = 0;
    
    // Get operation name
    virtual std::string name() const = 0;
    
    // Compute output shape from input shapes
    virtual std::vector<int> compute_output_shape(
        const std::vector<Tensor>& inputs) = 0;
};

class AddOp : public Operation {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
    std::string name() const override { return "AddOp"; }
    std::vector<int> compute_output_shape(const std::vector<Tensor>& inputs) override;
};

}  // namespace mlc
