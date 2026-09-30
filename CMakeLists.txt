cmake_minimum_required(VERSION 4.2)
project(gemmcuda CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(gemmcuda gemm.cu)

set_target_properties(gemmcuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
