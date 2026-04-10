#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include "core/tensor.h"
#include "core/operation.h"

namespace py = pybind11;
using namespace mlc;

PYBIND11_MODULE(mlc_core, m) {
    m.doc() = "MLC Tensor Compiler - CUDA-accelerated tensor operations";
    
    // Device enum
    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("CUDA", Device::CUDA)
        .export_values();
    
    // DataType enum
    py::enum_<DataType>(m, "DataType")
        .value("FLOAT32", DataType::FLOAT32)
        .value("FLOAT64", DataType::FLOAT64)
        .export_values();
    
    // Tensor class - comprehensive bindings
    py::class_<Tensor>(m, "Tensor")
        // Constructors
        .def(py::init<>(), "Create empty tensor")
        .def(py::init<std::vector<float>, std::vector<int>>(),
             py::arg("data"), py::arg("shape"),
             "Create tensor from data and shape")
        .def(py::init<std::vector<int>>(),
             py::arg("shape"),
             "Create zero-initialized tensor with given shape")
        .def(py::init<std::vector<int>, Device>(),
             py::arg("shape"), py::arg("device") = Device::CPU,
             "Create zero-initialized tensor on specified device")
        
        // Properties - shape and size info
        .def("shape", &Tensor::shape, 
             py::return_value_policy::copy,
             "Get tensor shape as list [d0, d1, ...]")
        .def("strides", &Tensor::strides,
             py::return_value_policy::copy,
             "Get tensor strides for memory layout")
        .def("size", &Tensor::size, "Get total number of elements")
        .def("rank", &Tensor::rank, "Get tensor rank (number of dimensions)")
        
        // Device and dtype info
        .def("device", &Tensor::device, "Get tensor device (CPU or CUDA)")
        .def("dtype", &Tensor::dtype, "Get tensor data type (FLOAT32 or FLOAT64)")
        .def("requires_grad", &Tensor::requires_grad, 
             "Check if tensor requires gradients (Phase 4 feature)")
        
        // Data conversion
        .def("to_vector", &Tensor::to_vector,
             "Convert tensor data to Python list")
        
        // Tensor operations
        .def("__add__", [](Tensor& a, Tensor& b) {
            AddOp op;
            auto result = op.forward({a, b});
            return result[0];
        }, py::arg("other"), 
           "Add two tensors element-wise with broadcasting")
        
        .def("add", [](Tensor& a, Tensor& b) {
            AddOp op;
            auto result = op.forward({a, b});
            return result[0];
        }, py::arg("other"),
           "Alias for __add__")
        
        // Shape manipulation
        .def("reshape", [](Tensor& self, std::vector<int> new_shape) {
            return self.reshape(new_shape);
        }, py::arg("new_shape"),
           "Reshape tensor to new shape (total size must match)")
        
        .def("flatten", [](Tensor& self) {
            return self.flatten();
        }, "Reshape tensor to 1D")
        
        // Device operations
        .def("to_device", [](Tensor& self, Device device) {
            return self.to_device(device);
        }, py::arg("device"),
           "Move tensor to specified device (CPU/CUDA)")
        
        .def("cpu", [](Tensor& self) {
            return self.to_device(Device::CPU);
        }, "Move tensor to CPU")
        
        .def("cuda", [](Tensor& self) {
            return self.to_device(Device::CUDA);
        }, "Move tensor to CUDA (Phase 3)")
        
        // Utilities and representations
        .def("__repr__", &Tensor::repr, "String representation of tensor")
        .def("__str__", &Tensor::repr, "String representation of tensor")
        .def("__len__", &Tensor::size, "Number of elements in tensor")
        
        // Comparison by value (for testing)
        .def("__eq__", [](const Tensor& a, const Tensor& b) {
            if (a.shape() != b.shape()) return false;
            if (a.size() != b.size()) return false;
            for (size_t i = 0; i < a.size(); ++i) {
                if (std::abs(a.data()[i] - b.data()[i]) > 1e-6f) return false;
            }
            return true;
        }, py::arg("other"),
           "Check tensor equality (with small tolerance for floats)");
    
    // Utility functions for creating tensors
    m.def("zeros", [](std::vector<int> shape, Device device = Device::CPU) {
        return Tensor(shape, device);
    }, py::arg("shape"), py::arg("device") = Device::CPU,
       "Create zero-initialized tensor");
    
    m.def("ones", [](std::vector<int> shape, Device device = Device::CPU) {
        Tensor t(shape, device);
        for (size_t i = 0; i < t.size(); ++i) {
            t.data()[i] = 1.0f;
        }
        return t;
    }, py::arg("shape"), py::arg("device") = Device::CPU,
       "Create one-initialized tensor");
    
    m.def("empty", [](std::vector<int> shape, Device device = Device::CPU) {
        return Tensor(shape, device);
    }, py::arg("shape"), py::arg("device") = Device::CPU,
       "Create empty tensor (uninitialized, same as zeros)");
    
    m.def("full", [](std::vector<int> shape, float value, Device device = Device::CPU) {
        Tensor t(shape, device);
        for (size_t i = 0; i < t.size(); ++i) {
            t.data()[i] = value;
        }
        return t;
    }, py::arg("shape"), py::arg("value"), py::arg("device") = Device::CPU,
       "Create tensor filled with constant value");
}
