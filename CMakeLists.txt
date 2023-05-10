cmake_minimum_required(VERSION 3.16)

project(tensor_core_gemm LANGUAGES CUDA CXX C)

# 设置 C++ 标准为 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDA REQUIRED)

# 设置 CUDA 标准为 C++17 ???????????????
# set(CMAKE_CUDA_STANDARD 17)
# set (CMAKE_CUDA_ARCHITECTURES 70)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_EXTENSIONS OFF)

add_executable(tensor_core_gemm_naive tensor_core_gemm_naive.cu)
target_link_libraries(tensor_core_gemm_naive ${CUDA_LIBRARIES})
target_include_directories(tensor_core_gemm_naive PUBLIC /home/data/cutlass/cutlass/build/install/include)
target_compile_options(tensor_core_gemm_naive PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_70 -std=c++17>)

# add_executable(tensor_core_gemm_advanced tensor_core_gemm_advanced.cu)
# target_link_libraries(tensor_core_gemm_advanced ${CUDA_LIBRARIES})
# target_include_directories(tensor_core_gemm_advanced PUBLIC /home/data/cutlass/cutlass/build/install/include)
# target_compile_options(tensor_core_gemm_advanced PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_70 -std=c++17>)
