#ifndef TENSOR_CORE_GEMM_NAIVE__H_
#define TENSOR_CORE_GEMM_NAIVE__H_

#include <mma.h>

#include <vector>


void 
gemm_gpu(
    const half *__restrict__ A, 
    const half *__restrict__ B, 
    float *__restrict__ C, 
    size_t M,
    size_t N, 
    size_t K);

std::vector<std::vector<float>> matmul(std::vector<std::vector<float>> & A, std::vector<std::vector<float>> & B);

#endif