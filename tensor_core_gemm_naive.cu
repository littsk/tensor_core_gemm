#include <mma.h>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include "cutlass/library/library.h"
#include "cutlass/library/handle.h"

#include <cuda_runtime.h>
#include <iostream>
#include <random>

#include "tensor_core_gemm_naive.h"

#define BMMA_M 32
#define BMMA_N 32
#define BMMA_K 32

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define WARP_SIZE 32

#define SCALE 2048.f

#define CUDACHECK(cmd)                                         \
    do                                                         \
    {                                                          \
        cudaError_t e = cmd;                                   \
        if (e != cudaSuccess)                                  \
        {                                                      \
            printf("Failed: Cuda error %s:%d '%s'\n",          \
                   __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

__global__ void
wmmaNaiveKernel(
    const half *__restrict__ A,
    const half *__restrict__ B,
    float *__restrict__ C,
    size_t M,
    size_t N,
    size_t K)
{
    // A row major
    // B row major
    extern __shared__ half shm[];
    half *a_blk_shm_ptr = shm;
    half *b_blk_shm_ptr = shm + BMMA_M * BMMA_K;
    float *c_blk_shm_ptr = (float *)(shm + BMMA_M * BMMA_K + BMMA_K * BMMA_N);

    // 全局的 warp idx
    // 一个warp负责一个C中的tile，一个C中的tile的大小是（WMMA_M，WMMA_N）
    size_t warp_tile_col_idx = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t warp_tile_row_idx = (blockIdx.y * blockDim.y + threadIdx.y); // C中 col_tile_idx

    size_t block_tile_col_idx = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
    size_t block_tile_row_idx = (blockIdx.y * blockDim.y + threadIdx.y) / blockDim.y;

    // block 中的 warp idx
    size_t warp_col_idx = threadIdx.x / WARP_SIZE;
    size_t warp_row_idx = threadIdx.y;
    // warp 中的 thread idx
    size_t threadID = (threadIdx.x + threadIdx.y * blockDim.x) % WARP_SIZE;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    for (size_t i = 0; i < K; i += BMMA_K)
    {
        size_t a_block_tile_left_idx = i;
        size_t a_block_tile_up_idx = block_tile_row_idx * BMMA_M;
        size_t b_block_tile_left_idx = block_tile_col_idx * BMMA_N;
        size_t b_block_tile_up_idx = i;

        // 一个warp负责搬运一个warp tile大小的数据

        size_t a_warp_tile_shm_start_idx = (warp_row_idx * WMMA_M) * BMMA_K + (warp_col_idx * WMMA_K);
        // warp tile 在 global中的坐标
        size_t a_warp_tile_up_idx = a_block_tile_up_idx + warp_row_idx * WMMA_M;
        size_t a_warp_tile_left_idx = a_block_tile_left_idx + warp_col_idx * WMMA_K;
        size_t a_warp_tile_start_idx = a_warp_tile_up_idx * K + a_warp_tile_left_idx;

        size_t b_warp_tile_shm_start_idx = (warp_row_idx * WMMA_K) * BMMA_N + (warp_col_idx * WMMA_N);
        // warp tile 在 global中的坐标
        size_t b_warp_tile_up_idx = b_block_tile_up_idx + warp_row_idx * WMMA_K;
        size_t b_warp_tile_left_idx = b_block_tile_left_idx + warp_col_idx * WMMA_N;
        size_t b_warp_tile_start_idx = b_warp_tile_up_idx * N + b_warp_tile_left_idx;

        if (threadID < WMMA_K)
        { // threadID 应该严格小于WMMA_K
            for (int j = 0; j < WMMA_M; ++j)
            {
                if ((a_warp_tile_up_idx + j) < M && (a_warp_tile_left_idx + threadID) < K)
                {
                    *(a_blk_shm_ptr + a_warp_tile_shm_start_idx + (j * BMMA_K) + threadID) =
                        *(A + a_warp_tile_start_idx + (j * K) + threadID);
                }
                else
                {
                    // printf("dive into bound a\n");
                    *(a_blk_shm_ptr + a_warp_tile_shm_start_idx + (j * BMMA_K) + threadID) = 0_hf;
                }
            }
        }
        if (threadID < WMMA_N)
        { 
            for (int j = 0; j < WMMA_K; ++j) 
            {
                if ((b_warp_tile_up_idx + j) < N && (b_warp_tile_left_idx + threadID) < K)
                {
                    *(b_blk_shm_ptr + b_warp_tile_shm_start_idx + (j * BMMA_N) + threadID) =
                        *(B + b_warp_tile_start_idx + (j * N) + threadID);
                }
                else
                {
                    // printf("dive into bound b col : %lld, row : %lld\n", b_block_tile_left_idx + i, b_blocktile_up_idxw + threadID);
                    *(b_blk_shm_ptr + b_warp_tile_shm_start_idx + (j * BMMA_N) + threadID) = 0_hf;
                }
            }
        }
        __syncthreads();

        for (int k = 0; k < BMMA_K; k += WMMA_K)
        {
            if (warp_tile_row_idx == 0 && warp_tile_col_idx == 0 && threadID == 0)
            {
                printf("yes\n");
            }
            if (a_block_tile_up_idx < M && a_block_tile_left_idx < K && b_block_tile_up_idx < K && b_block_tile_left_idx < N)
            { // 这个判断是有必要的吗
                nvcuda::wmma::load_matrix_sync(a_frag, a_blk_shm_ptr + (warp_row_idx * WMMA_M) * BMMA_K + k, BMMA_K);
                nvcuda::wmma::load_matrix_sync(b_frag, b_blk_shm_ptr + k * BMMA_K + (warp_col_idx * WMMA_N), BMMA_N);
                nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
        }
        __syncthreads();
    }

    // 一个warp负责把一个warp tile的结果写回 shared memory 和 global memory
    size_t warp_tile_left_idx = warp_tile_col_idx * WMMA_N;
    size_t warp_tile_up_idx = warp_tile_row_idx * WMMA_M;
    if (warp_tile_up_idx < M && warp_tile_left_idx < N) // 这个判断是有必要的嘛
    { 
        size_t c_frag_shm_start_idx = (warp_row_idx * WMMA_M) * BMMA_N + (warp_col_idx * WMMA_N);
        size_t c_frag_global_start_idx = warp_tile_up_idx * N + warp_tile_left_idx;

        nvcuda::wmma::store_matrix_sync(c_blk_shm_ptr + c_frag_shm_start_idx, c_frag, BMMA_N, nvcuda::wmma::mem_row_major);

        if (threadID < WMMA_N)
        {
            for (int i = 0; i < WMMA_M; ++i)
            {
                if (warp_tile_up_idx + i < M && warp_tile_left_idx + threadID < N)
                {
                    *(C + c_frag_global_start_idx + (i * N) + threadID) =
                        *(c_blk_shm_ptr + c_frag_shm_start_idx + (i * BMMA_N) + threadID);
                }
            }
        }
    }
    // if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0){
    //     printf("%f\n", __half2float(a_frag.x[1]));
    // }
}

void gemm_gpu(
    const half *__restrict__ A,
    const half *__restrict__ B,
    float *__restrict__ C,
    size_t M,
    size_t N,
    size_t K)
{
    dim3 block(BMMA_M / WMMA_M * WARP_SIZE, BMMA_N / WMMA_N);
    dim3 grid((M - 1) / (WMMA_M * block.x / WARP_SIZE) + 1, (N - 1) / (WMMA_N * block.y) + 1);

    size_t shm_size = (BMMA_M * BMMA_K + BMMA_N * BMMA_K) * sizeof(half) + BMMA_M * BMMA_N * sizeof(float);
    std::cout << grid.x << " " << grid.y << std::endl;
    wmmaNaiveKernel<<<grid, block, shm_size>>>(A, B, C, M, N, K);
}

std::vector<std::vector<float>>
matmul(
    std::vector<std::vector<float>> &A,
    std::vector<std::vector<float>> &B)
{
    size_t M = A.size(), K = A[0].size(), N = B[0].size();
    return {{}};
}

void gemm_cpu(
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *__restrict__ C,
    size_t M,
    size_t N,
    size_t K)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            int c_idx = i * N + j;
            for (int k = 0; k < K; ++k)
            {
                int a_idx = i * K + k;
                int b_idx = k * K + j;
                C[c_idx] += A[a_idx] * B[b_idx];
            }
        }
    }
}

int cutlass_gemm(
    cutlass::half_t *__restrict__ A,
    cutlass::half_t *__restrict__ B,
    float *__restrict__ C,
    const int M,
    const int N,
    const int K)
{

    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t,
        cutlass::layout::RowMajor,
        cutlass::half_t,
        cutlass::layout::RowMajor,
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassWmmaTensorOp,
        cutlass::arch::Sm70,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<32, 32, 32>,
        cutlass::gemm::GemmShape<16, 16, 16>>;

    float alpha = 1.f;
    float beta = 0.f;

    Gemm gemm_op;
    cutlass::Status status;

    status = gemm_op({{M, N, K},
                      {A, K},
                      {B, K},
                      {C, N},
                      {C, N},
                      {alpha, beta}});

    if (status != cutlass::Status::kSuccess)
    {
        return -1;
    }

    return 0;
}

int main(int agrc, char *argv[])
{
    std::srand(320);
    int m = 63, n = 63, k = 63;

    float *ha, *hb, *hc;
    ha = (float *)malloc(m * k * sizeof(float));
    hb = (float *)malloc(k * n * sizeof(float));
    hc = (float *)malloc(m * n * sizeof(float));

    half *da, *db;
    float *dc;
    CUDACHECK(cudaMallocManaged(&da, m * k * sizeof(half)));
    CUDACHECK(cudaMallocManaged(&db, k * n * sizeof(half)));
    CUDACHECK(cudaMallocManaged(&dc, m * n * sizeof(float)));

    cutlass::half_t *cut_da, *cut_db;
    float *cut_dc;
    CUDACHECK(cudaMallocManaged(&cut_da, m * k * sizeof(cutlass::half_t)));
    CUDACHECK(cudaMallocManaged(&cut_db, k * n * sizeof(cutlass::half_t)));
    CUDACHECK(cudaMallocManaged(&cut_dc, m * n * sizeof(float)));

    for (int i = 0; i < m * k; ++i)
    {
        ha[i] = i / SCALE;
        da[i] = ha[i];
        cut_da[i] = ha[i];
    }
    for (int i = 0; i < k * n; ++i)
    {
        hb[i] = i / SCALE;
        db[i] = hb[i];
        cut_db[i] = hb[i];
    }
    memset(hc, 0, m * n * sizeof(float));
    cudaMemset(dc, 0, m * n * sizeof(float));
    cudaMemset(cut_dc, 0, m * n * sizeof(float));

    gemm_cpu(ha, hb, hc, m, n, k);
    gemm_gpu(da, db, dc, m, n, k);
    // cutlass_gemm(cut_da, cut_db, cut_dc, m, n, k);

    CUDACHECK(cudaDeviceSynchronize());

    bool flag = true;
    for (int i = 0; i < m * n; ++i)
    {
        if (abs((hc[i] - dc[i]) / max(hc[i], dc[i])) > 1e-3)
        { // || (hc[i] - cut_dc[i]) > 1e-5){
            printf("%d error: hc %f, dc %f, cut_dc %f\n", i, hc[i], dc[i], cut_dc[i]);
            flag = false;
            break;
        }
    }
    if (flag)
    {
        printf("all close\n");
    }
    printf("test: hc %f, dc %f, cut_dc %f\n", hc[32], dc[32], cut_dc[32]);

    CUDACHECK(cudaFree(da));
    CUDACHECK(cudaFree(db));
    CUDACHECK(cudaFree(dc));
    CUDACHECK(cudaFree(cut_da));
    CUDACHECK(cudaFree(cut_db));
    CUDACHECK(cudaFree(cut_dc));
    free(ha);
    free(hb);
    free(hc);
    return 0;
}