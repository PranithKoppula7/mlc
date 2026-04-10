#include "../core/operation.h"
#include "../core/broadcast.h"
#include "../runtime/executor.h"
#include <stdexcept>

namespace mlc {

std::vector<Tensor> AddOp::forward(const std::vector<Tensor>& inputs) {
    if (inputs.size() != 2) {
        throw std::invalid_argument("AddOp requires exactly 2 inputs");
    }
    
    const Tensor& a = inputs[0];
    const Tensor& b = inputs[1];
    
    // Validate devices match
    if (a.device() != b.device()) {
        throw std::invalid_argument("Tensors must be on same device");
    }
    
    // Compute output shape via broadcasting
    std::vector<int> output_shape = compute_output_shape(inputs);
    
    // Create output tensor
    Tensor output(output_shape, a.device());
    
    // Execute addition
    Executor::execute_add(a, b, output);
    
    return {output};
}

std::vector<Tensor> AddOp::backward(const std::vector<Tensor>& grad_outputs) {
    if (grad_outputs.size() != 1) {
        throw std::invalid_argument("AddOp backward requires 1 gradient");
    }
    
    // For addition: grad_a = grad_out, grad_b = grad_out
    // With broadcasting, we need to reduce along broadcast dimensions
    std::vector<Tensor> grad_inputs = {grad_outputs[0], grad_outputs[0]};
    
    return grad_inputs;
}

std::vector<int> AddOp::compute_output_shape(const std::vector<Tensor>& inputs) {
    if (inputs.size() != 2) {
        throw std::invalid_argument("AddOp requires exactly 2 inputs");
    }
    
    return BroadcastResolver::compute_output_shape(
        inputs[0].shape(),
        inputs[1].shape()
    );
}

}  // namespace mlc
