# Code generation library for MLC tensor compiler
# Implements LLVM-based IR generation and JIT compilation

# Add LLVM components
find_package(LLVM 14 REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVM version: ${LLVM_VERSION}")
message(STATUS "LLVM lib dir: ${LLVM_LIBRARY_DIR}")

# Use monolithic LLVM library or map components
set(llvm_libs LLVM)

# Create codegen library
add_library(mlc_codegen
  llvm_context.cpp
  llvm_context.h
  llvm_module_builder.cpp
  llvm_module_builder.h
  ir_generator.cpp
  ir_generator.h
  jit_compiler.cpp
  jit_compiler.h
  kernel_cache.cpp
  kernel_cache.h
)

# Enable Position Independent Code for linking into Python module
set_target_properties(mlc_codegen PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Set C++ standard
target_compile_features(mlc_codegen PRIVATE cxx_std_17)

# Include LLVM headers
target_include_directories(mlc_codegen
  PRIVATE ${LLVM_INCLUDE_DIRS}
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Link against LLVM libraries
target_link_libraries(mlc_codegen
  PUBLIC ${llvm_libs}
  PRIVATE mlc_core
)

# Add LLVM definitions
target_compile_definitions(mlc_codegen PRIVATE ${LLVM_DEFINITIONS})

# Enable optimizations
target_compile_options(mlc_codegen PRIVATE -O2)
