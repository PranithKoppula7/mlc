cmake_minimum_required(VERSION 3.10)
project(mlc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find LLVM for Phase 2 code generation
find_package(LLVM 14 REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")

# Core library
add_library(mlc_core
    src/core/tensor.cpp
    src/core/tensor.h
    src/core/broadcast.cpp
    src/core/broadcast.h
    src/core/operation.cpp
    src/core/operation.h
    src/runtime/executor.cpp
    src/runtime/executor.h
    src/runtime/jit_executor.cpp
    src/runtime/jit_executor.h
)

# Enable PIC for static library so it can be linked into shared modules
set_target_properties(mlc_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(mlc_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Include LLVM headers (needed by jit_executor.h)
target_include_directories(mlc_core PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(mlc_core PRIVATE ${LLVM_DEFINITIONS})

# Code generation library (Phase 2)
add_subdirectory(src/codegen)
target_link_libraries(mlc_core PUBLIC mlc_codegen)

# Tests
enable_testing()

add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor mlc_core)
add_test(NAME TensorTest COMMAND test_tensor)

add_executable(test_broadcast tests/test_broadcast.cpp)
target_link_libraries(test_broadcast mlc_core)
add_test(NAME BroadcastTest COMMAND test_broadcast)

add_executable(test_add tests/test_add.cpp)
target_link_libraries(test_add mlc_core)
add_test(NAME AddTest COMMAND test_add)

add_executable(test_jit_executor tests/test_jit_executor.cpp)
target_link_libraries(test_jit_executor mlc_core)
add_test(NAME JITExecutorTest COMMAND test_jit_executor)

# Python bindings
find_package(pybind11 CONFIG)
if(pybind11_FOUND)
    message(STATUS "pybind11 found at ${pybind11_DIR}, building Python bindings")
    
    # Create Python module - uses pybind11's own add_module function
    pybind11_add_module(mlc_core_py python/tensor_py.cpp)
    
    # Link to our core library
    target_link_libraries(mlc_core_py PRIVATE mlc_core)
    
    # Add linker flags to handle LLVM relocations
    target_link_options(mlc_core_py PRIVATE -Wl,--allow-shlib-undefined)
    
    # Set output directory for the built module
    set_target_properties(mlc_core_py PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        LIBRARY_OUTPUT_NAME "mlc_core"
        PREFIX ""
    )
    
    # Add RPATH for local development so module can find dependencies
    set_target_properties(mlc_core_py PROPERTIES
        BUILD_RPATH "${CMAKE_BINARY_DIR}"
    )
    
    # Create mlc package directory and copy __init__.py
    add_custom_command(TARGET mlc_core_py POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/lib/mlc"
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/python/__init__.py" "${CMAKE_BINARY_DIR}/lib/mlc/__init__.py"
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_BINARY_DIR}/lib/mlc_core.cpython-312-x86_64-linux-gnu.so" "${CMAKE_BINARY_DIR}/lib/mlc/"
        COMMAND touch "${CMAKE_BINARY_DIR}/lib/mlc/__init__.py"
        COMMENT "Setting up mlc Python package..."
    )
    
    message(STATUS "Python module will be built as: ${CMAKE_BINARY_DIR}/lib/mlc/")
    message(STATUS "To use: export PYTHONPATH=${CMAKE_BINARY_DIR}/lib:$PYTHONPATH")
else()
    message(WARNING "pybind11 not found. Python bindings will not be built.")
    message(WARNING "To enable: pip install pybind11")
endif()
