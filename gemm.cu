#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <string>
#include <algorithm>

namespace wmma = nvcuda::wmma;

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << "\n"; std::exit(EXIT_FAILURE); } } while (0)
#define CUBLAS_CHECK(call) do { cublasStatus_t status = (call); if (status != CUBLAS_STATUS_SUCCESS) { std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << " status=" << status << std::endl; std::exit(EXIT_FAILURE); } } while (0)

constexpr int W = 32;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

struct BenchmarkResult {
    std::string name;
    float ms;
    double gflops;
    double error;
};

double max_error(const std::vector<float> &C, const std::vector<float> &C_ref) {
    double error = 0.0;
    for (size_t i = 0; i < C.size(); ++i) error = std::max(error, std::abs(static_cast<double>(C[i]) - static_cast<double>(C_ref[i])));
    return error;
}

template<typename KernelLauncher>
BenchmarkResult benchmark(const std::string &name, KernelLauncher kernelLauncher, float *d_C, size_t bytes_C, std::vector<float> &C, const std::vector<float> &C_ref, int M, int N, int K, int warmup = 2, int iterations = 2) {
    for (int r = 0; r < warmup; ++r) {
        CUDA_CHECK(cudaMemset(d_C, 0, bytes_C));
        kernelLauncher();
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int r = 0; r < iterations; ++r) {
        CUDA_CHECK(cudaMemset(d_C, 0, bytes_C));
        kernelLauncher();
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float ms = total_ms / iterations;

    CUDA_CHECK(cudaMemcpy(C.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));

    double error = max_error(C, C_ref);
    double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    double gflops = flops / (ms * 1.0e6);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return {name, ms, gflops, error};
}


__global__ void gemm_ground_truth(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int M, int N, int K) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= M || j >= N) return;

    float c_ij = 0.0f;
    for (int k = 0; k < K; ++k) c_ij += A[i * K + k] * B[k * N + j];

    C[i * N + j] = c_ij;
}


// ============================================================
// 1. Global memory + bad/non-coalesced mapping
// Consecutive x lanes move through rows.
// ============================================================

__global__ void gemm_global_bad(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int M, int N, int K) {
    int i = threadIdx.x;
    int j = threadIdx.y;

    if (i >= M || j >= N) return;

    float c_ij = 0.0f;
    for (int k = 0; k < K; ++k) c_ij += A[i * K + k] * B[k * N + j];

    C[i * N + j] = c_ij;
}


// ============================================================
// 2. Global memory + coalesced mapping
// At fixed k, C[i,j:j+W] and B[k,j:j+W] are contiguous.
// ============================================================

__global__ void gemm_global_good(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int M, int N, int K) {
    int j = threadIdx.x;
    int i = threadIdx.y;

    if (i >= M || j >= N) return;

    float c_ij = 0.0f;
    for (int k = 0; k < K; ++k) c_ij += A[i * K + k] * B[k * N + j];

    C[i * N + j] = c_ij;
}


// ============================================================
// 3. Block tiling + global memory
// i = blockIdx.y * blockDim.y + threadIdx.y
// j = blockIdx.x * blockDim.x + threadIdx.x
// ============================================================

template<int BLOCK_M, int BLOCK_N>
__global__ void gemm_block_global(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int M, int N, int K) {
    int i = blockIdx.y * BLOCK_M + threadIdx.y;
    int j = blockIdx.x * BLOCK_N + threadIdx.x;

    if (i >= M || j >= N) return;

    float c_ij = 0.0f;
    for (int k = 0; k < K; ++k) c_ij += A[i * K + k] * B[k * N + j];

    C[i * N + j] = c_ij;
}


// ============================================================
// 4. Block tiling + shared memory
//
// A_block : BLOCK_M x k_c
// B_block : k_c x BLOCK_N
//
// k_tile_start = 0, k_c, 2*k_c, ...
// ============================================================

template<int BLOCK_M, int BLOCK_N, int k_c>
__global__ void gemm_block_shared(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int M, int N, int K) {
    static_assert(k_c <= BLOCK_N, "k_c must be <= BLOCK_N.");
    static_assert(k_c <= BLOCK_M, "k_c must be <= BLOCK_M.");

    __shared__ float A_block[BLOCK_M][k_c];
    __shared__ float B_block[k_c][BLOCK_N];

    int row = threadIdx.y;
    int col = threadIdx.x;

    int i = blockIdx.y * BLOCK_M + row;
    int j = blockIdx.x * BLOCK_N + col;

    float c_ij = 0.0f;

    for (int k_tile_start = 0; k_tile_start < K; k_tile_start += k_c) {
        if (col < k_c) {
            int k = k_tile_start + col;
            A_block[row][col] = (i < M && k < K) ? A[i * K + k] : 0.0f;
        }

        if (row < k_c) {
            int k = k_tile_start + row;
            B_block[row][col] = (k < K && j < N) ? B[k * N + j] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < k_c; ++kk) c_ij += A_block[row][kk] * B_block[kk][col];

        __syncthreads();
    }

    if (i < M && j < N) C[i * N + j] = c_ij;
}


// ============================================================
// 5. Block + shared memory + warp tiling
//
// tid    = threadIdx.x
// warpId = tid / W
//
// warpRow = warpId / (BLOCK_N / N_W)
// warpCol = warpId % (BLOCK_N / N_W)
//
// i = blockIdx.y * BLOCK_M + M_W * warpRow + (tid % W) / N_W
// j = blockIdx.x * BLOCK_N + N_W * warpCol + (tid % W) % N_W
// ============================================================

template<int BLOCK_M, int BLOCK_N, int k_c, int M_W, int N_W>
__global__ void gemm_warp_tiled(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int M, int N, int K) {
    static_assert(M_W * N_W == W, "M_W * N_W must equal warp size.");
    static_assert(BLOCK_M % M_W == 0, "BLOCK_M must be divisible by M_W.");
    static_assert(BLOCK_N % N_W == 0, "BLOCK_N must be divisible by N_W.");

    constexpr int WARPS = (BLOCK_M / M_W) * (BLOCK_N / N_W);
    constexpr int T = WARPS * W;

    static_assert(T <= 1024, "Too many threads per CUDA block.");

    __shared__ float A_block[BLOCK_M][k_c];
    __shared__ float B_block[k_c][BLOCK_N];

    int tid = threadIdx.x;
    int warpId = tid / W;

    int warpRow = warpId / (BLOCK_N / N_W);
    int warpCol = warpId % (BLOCK_N / N_W);

    int row = M_W * warpRow + (tid % W) / N_W;
    int col = N_W * warpCol + (tid % W) % N_W;

    int i = blockIdx.y * BLOCK_M + row;
    int j = blockIdx.x * BLOCK_N + col;

    float c_ij = 0.0f;

    for (int k_tile_start = 0; k_tile_start < K; k_tile_start += k_c) {

        // Cooperative population of A_block (cf readme)
        for (int idx = tid; idx < BLOCK_M * k_c; idx += T) {
            int sharedRow = idx / k_c;
            int sharedCol = idx % k_c;
            int globalRow = blockIdx.y * BLOCK_M + sharedRow;
            int k = k_tile_start + sharedCol;

            A_block[sharedRow][sharedCol] = (globalRow < M && k < K) ? A[globalRow * K + k] : 0.0f;
        }

        // Cooperative population of B_block (cf readme)
        for (int idx = tid; idx < k_c * BLOCK_N; idx += T) {
            int sharedRow = idx / BLOCK_N;
            int sharedCol = idx % BLOCK_N;
            int k = k_tile_start + sharedRow;
            int globalCol = blockIdx.x * BLOCK_N + sharedCol;

            B_block[sharedRow][sharedCol] = (k < K && globalCol < N) ? B[k * N + globalCol] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < k_c; ++kk) c_ij += A_block[row][kk] * B_block[kk][col];

        __syncthreads();
    }

    if (i < M && j < N) C[i * N + j] = c_ij;
}


// ============================================================
// 6. Block + shared + warp + register tiling
//
// One warp computes M_W x N_W.
// Each lane computes M_R x N_R.
//
// microRow = (tid % W) / (N_W / N_R)
// microCol = (tid % W) % (N_W / N_R)
//
// i = blockIdx.y*BLOCK_M + warpRow*M_W + microRow*M_R + r
// j = blockIdx.x*BLOCK_N + warpCol*N_W + microCol*N_R + c
// ============================================================

template<int BLOCK_M, int BLOCK_N, int k_c, int M_W, int N_W, int M_R, int N_R>
__global__ void gemm_register_tiled(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int M, int N, int K) {
    static_assert(BLOCK_M % M_W == 0, "BLOCK_M must be divisible by M_W.");
    static_assert(BLOCK_N % N_W == 0, "BLOCK_N must be divisible by N_W.");
    static_assert(M_W % M_R == 0, "M_W must be divisible by M_R.");
    static_assert(N_W % N_R == 0, "N_W must be divisible by N_R.");
    static_assert((M_W / M_R) * (N_W / N_R) == W, "Warp tile must map to exactly 32 thread microtiles.");

    constexpr int WARPS = (BLOCK_M / M_W) * (BLOCK_N / N_W);
    constexpr int THREADS = WARPS * W;

    static_assert(THREADS <= 1024, "Too many threads per CUDA block.");

    __shared__ float A_block[BLOCK_M][k_c];
    __shared__ float B_block[k_c][BLOCK_N];

    int tid = threadIdx.x;
    int warpId = tid / W;

    int warpRow = warpId / (BLOCK_N / N_W);
    int warpCol = warpId % (BLOCK_N / N_W);

    int microRow = (tid % W) / (N_W / N_R);
    int microCol = (tid % W) % (N_W / N_R);

    float C_R[M_R][N_R]; // C is resident here

    #pragma unroll
    for (int r = 0; r < M_R; ++r)
        #pragma unroll
        for (int c = 0; c < N_R; ++c) C_R[r][c] = 0.0f;

    for (int k_tile_start = 0; k_tile_start < K; k_tile_start += k_c) {

        // Cooperative population of A_block (cf readme)
        for (int idx = tid; idx < BLOCK_M * k_c; idx += THREADS) {
            int row = idx / k_c;
            int col = idx % k_c;
            int i = blockIdx.y * BLOCK_M + row;
            int k = k_tile_start + col;

            A_block[row][col] = (i < M && k < K) ? A[i * K + k] : 0.0f;
        }

        // Cooperative population of B_block (cf readme)
        for (int idx = tid; idx < k_c * BLOCK_N; idx += THREADS) {
            int row = idx / BLOCK_N;
            int col = idx % BLOCK_N;
            int k = k_tile_start + row;
            int j = blockIdx.x * BLOCK_N + col;

            B_block[row][col] = (k < K && j < N) ? B[k * N + j] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < k_c; ++kk) {
            float A_R[M_R];
            float B_R[N_R];

            #pragma unroll
            for (int r = 0; r < M_R; ++r) A_R[r] = A_block[warpRow * M_W + microRow * M_R + r][kk];

            #pragma unroll
            for (int c = 0; c < N_R; ++c) B_R[c] = B_block[kk][warpCol * N_W + microCol * N_R + c];

            // outer product
            #pragma unroll
            for (int r = 0; r < M_R; ++r)
                #pragma unroll
                for (int c = 0; c < N_R; ++c) C_R[r][c] += A_R[r] * B_R[c];
        }

        __syncthreads();
    }


    // (cf readme to better understand mapping to tensor locations...)
    #pragma unroll
    for (int r = 0; r < M_R; ++r) {
        #pragma unroll
        for (int c = 0; c < N_R; ++c) {
            int i = blockIdx.y * BLOCK_M + warpRow * M_W + microRow * M_R + r;
            int j = blockIdx.x * BLOCK_N + warpCol * N_W + microCol * N_R + c;

            if (i < M && j < N) C[i * N + j] = C_R[r][c];
        }
    }
}


// ============================================================
// 7. Block + shared + warp + Tensor Cores
//
// One warp computes one WMMA_M x WMMA_N output tile.
//
// warpRow = warpId / (BLOCK_N / WMMA_N)
// warpCol = warpId % (BLOCK_N / WMMA_N)
// ============================================================

template<int BLOCK_M, int BLOCK_N, int k_c>
__global__ void gemm_tensor_core_shared(const half *__restrict__ A, const half *__restrict__ B, float *__restrict__ C, int M, int N, int K) {
    static_assert(BLOCK_M % WMMA_M == 0, "BLOCK_M must be divisible by WMMA_M.");
    static_assert(BLOCK_N % WMMA_N == 0, "BLOCK_N must be divisible by WMMA_N.");
    static_assert(k_c % WMMA_K == 0, "k_c must be divisible by WMMA_K.");

    constexpr int WARPS = (BLOCK_M / WMMA_M) * (BLOCK_N / WMMA_N);
    constexpr int THREADS = WARPS * W;

    //  half-precision for input fragments 𝐴 and B
    __shared__ half A_block[BLOCK_M][k_c];
    __shared__ half B_block[k_c][BLOCK_N];

    int tid = threadIdx.x;
    int warpId = tid / W;

    if (warpId >= WARPS) return;

    int warpRow = warpId / (BLOCK_N / WMMA_N);
    int warpCol = warpId % (BLOCK_N / WMMA_N);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k_tile_start = 0; k_tile_start < K; k_tile_start += k_c) {

        for (int idx = tid; idx < BLOCK_M * k_c; idx += THREADS) {
            int row = idx / k_c;
            int col = idx % k_c;
            int i = blockIdx.y * BLOCK_M + row;
            int k = k_tile_start + col;

            A_block[row][col] = (i < M && k < K) ? A[i * K + k] : __float2half(0.0f);
        }

        for (int idx = tid; idx < k_c * BLOCK_N; idx += THREADS) {
            int row = idx / BLOCK_N;
            int col = idx % BLOCK_N;
            int k = k_tile_start + row;
            int j = blockIdx.x * BLOCK_N + col;

            B_block[row][col] = (k < K && j < N) ? B[k * N + j] : __float2half(0.0f);
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < k_c; kk += WMMA_K) {
            const half *a = &A_block[warpRow * WMMA_M][kk];
            const half *b = &B_block[kk][warpCol * WMMA_N];

            wmma::load_matrix_sync(a_frag, a, k_c);
            wmma::load_matrix_sync(b_frag, b, BLOCK_N);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }

        __syncthreads();
    }

    int i = blockIdx.y * BLOCK_M + warpRow * WMMA_M;
    int j = blockIdx.x * BLOCK_N + warpCol * WMMA_N;

    if (i + WMMA_M <= M && j + WMMA_N <= N) wmma::store_matrix_sync(C + i * N + j, acc_frag, N, wmma::mem_row_major);
}

__global__ void float_to_half(const float *input, half *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = __float2half(input[idx]);
}


void print_header() {
    std::cout << std::left << std::setw(60) << "Kernel"
              << std::right << std::setw(14) << "Time(ms)"
              << std::setw(16) << "GFLOP/s"
              << std::setw(18) << "Max Error"
              << std::setw(10) << "Status" << "\n";

    std::cout << std::string(118, '-') << "\n";
}

void print_result(const BenchmarkResult &result, double tolerance) {
    std::cout << std::left << std::setw(60) << result.name
              << std::right << std::setw(14) << std::fixed << std::setprecision(4) << result.ms
              << std::setw(16) << std::fixed << std::setprecision(2) << result.gflops
              << std::setw(18) << std::scientific << std::setprecision(3) << result.error
              << std::setw(10) << (result.error <= tolerance ? "PASS" : "FAIL") << "\n";
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "\nGPU: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Warp size: " << prop.warpSize << "\n\n";

    cublasHandle_t cublasHandle;
    CUBLAS_CHECK(cublasCreate(&cublasHandle));


    // ========================================================
    // PART A: SMALL MATRIX
    // ========================================================

    constexpr int SMALL_M = 32;
    constexpr int SMALL_N = 32;
    constexpr int SMALL_K = 512;

    std::cout << "============================================================\n"
              << "PART A: SMALL MATRIX -- ONE BLOCK\n"
              << "M = " << SMALL_M << ", N = " << SMALL_N << ", K = " << SMALL_K << "\n"
              << "============================================================\n\n";

    std::vector<float> A_small(SMALL_M * SMALL_K);
    std::vector<float> B_small(SMALL_K * SMALL_N);
    std::vector<float> C_small(SMALL_M * SMALL_N);
    std::vector<float> C_small_ref(SMALL_M * SMALL_N);

    std::srand(0);
    for (float &x : A_small) x = (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 0.1f;
    for (float &x : B_small) x = (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 0.1f;

    float *d_A_small, *d_B_small, *d_C_small, *d_C_small_ref;

    size_t bytes_A_small = A_small.size() * sizeof(float);
    size_t bytes_B_small = B_small.size() * sizeof(float);
    size_t bytes_C_small = C_small.size() * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_A_small, bytes_A_small));
    CUDA_CHECK(cudaMalloc(&d_B_small, bytes_B_small));
    CUDA_CHECK(cudaMalloc(&d_C_small, bytes_C_small));
    CUDA_CHECK(cudaMalloc(&d_C_small_ref, bytes_C_small));

    CUDA_CHECK(cudaMemcpy(d_A_small, A_small.data(), bytes_A_small, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_small, B_small.data(), bytes_B_small, cudaMemcpyHostToDevice));

    dim3 refBlock(16, 16);
    dim3 refGridSmall((SMALL_N + 15) / 16, (SMALL_M + 15) / 16);

    gemm_ground_truth<<<refGridSmall, refBlock>>>(d_A_small, d_B_small, d_C_small_ref, SMALL_M, SMALL_N, SMALL_K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C_small_ref.data(), d_C_small_ref, bytes_C_small, cudaMemcpyDeviceToHost));

    dim3 block_bad(SMALL_M, SMALL_N);

    BenchmarkResult result_bad = benchmark(
        "1. Global memory + bad/non-coalesced mapping",
        [&]() { gemm_global_bad<<<1, block_bad>>>(d_A_small, d_B_small, d_C_small, SMALL_M, SMALL_N, SMALL_K); },
        d_C_small, bytes_C_small, C_small, C_small_ref, SMALL_M, SMALL_N, SMALL_K
    );

    dim3 block_good(SMALL_N, SMALL_M);

    BenchmarkResult result_good = benchmark(
        "2. Global memory + coalesced mapping",
        [&]() { gemm_global_good<<<1, block_good>>>(d_A_small, d_B_small, d_C_small, SMALL_M, SMALL_N, SMALL_K); },
        d_C_small, bytes_C_small, C_small, C_small_ref, SMALL_M, SMALL_N, SMALL_K
    );

    print_header();
    print_result(result_bad, 1e-3);
    print_result(result_good, 1e-3);

    CUDA_CHECK(cudaFree(d_A_small));
    CUDA_CHECK(cudaFree(d_B_small));
    CUDA_CHECK(cudaFree(d_C_small));
    CUDA_CHECK(cudaFree(d_C_small_ref));


    // ========================================================
    // PART B: LARGE MATRIX
    // ========================================================

    constexpr int M = 2048;
    constexpr int N = 2048;
    constexpr int K = 2048;

    std::cout << "\n\n============================================================\n"
              << "PART B: LARGE MATRIX\n"
              << "M = " << M << ", N = " << N << ", K = " << K << "\n"
              << "============================================================\n\n";

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N);
    std::vector<float> C_ref(M * N);

    std::srand(1);
    for (float &x : A) x = (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 0.1f;
    for (float &x : B) x = (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 0.1f;

    float *d_A, *d_B, *d_C, *d_C_ref;

    size_t bytes_A = A.size() * sizeof(float);
    size_t bytes_B = B.size() * sizeof(float);
    size_t bytes_C = C.size() * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
    CUDA_CHECK(cudaMalloc(&d_C_ref, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), bytes_B, cudaMemcpyHostToDevice));


    // Ground truth

    std::cout << "Computing GPU FP32 ground truth...\n";

    dim3 refGrid((N + 15) / 16, (M + 15) / 16);

    gemm_ground_truth<<<refGrid, refBlock>>>(d_A, d_B, d_C_ref, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C_ref.data(), d_C_ref, bytes_C, cudaMemcpyDeviceToHost));

    std::cout << "Ground truth complete.\n\n";

    std::vector<BenchmarkResult> results;


    // ========================================================
    // 3. BLOCK TILING + GLOBAL MEMORY
    // ========================================================

    constexpr int BG_M = 16;
    constexpr int BG_N = 16;

    dim3 block_global(BG_N, BG_M);
    dim3 grid_global((N + BG_N - 1) / BG_N, (M + BG_M - 1) / BG_M);

    results.push_back(benchmark(
        "3. Block tiling + global memory",
        [&]() { gemm_block_global<BG_M, BG_N><<<grid_global, block_global>>>(d_A, d_B, d_C, M, N, K); },
        d_C, bytes_C, C, C_ref, M, N, K
    ));


    // ========================================================
    // 4. BLOCK TILING + SHARED MEMORY
    // ========================================================

    constexpr int BS_M = 16;
    constexpr int BS_N = 16;
    constexpr int BS_K = 16;

    dim3 block_shared(BS_N, BS_M);
    dim3 grid_shared((N + BS_N - 1) / BS_N, (M + BS_M - 1) / BS_M);

    results.push_back(benchmark(
        "4. Block tiling + shared memory",
        [&]() { gemm_block_shared<BS_M, BS_N, BS_K><<<grid_shared, block_shared>>>(d_A, d_B, d_C, M, N, K); },
        d_C, bytes_C, C, C_ref, M, N, K
    ));


    // ========================================================
    // 5. BLOCK + SHARED + WARP TILING
    // ========================================================

    constexpr int WT_BLOCK_M = 16;
    constexpr int WT_BLOCK_N = 32;
    constexpr int WT_K = 16;

    constexpr int WT_M_W = 4;
    constexpr int WT_N_W = 8;

    static_assert(WT_M_W * WT_N_W == W, "One warp tile must contain exactly 32 outputs.");

    constexpr int WT_WARPS = (WT_BLOCK_M / WT_M_W) * (WT_BLOCK_N / WT_N_W);
    constexpr int WT_THREADS = WT_WARPS * W;

    dim3 grid_warp((N + WT_BLOCK_N - 1) / WT_BLOCK_N, (M + WT_BLOCK_M - 1) / WT_BLOCK_M);

    results.push_back(benchmark(
        "5. Block + shared + warp tiling",
        [&]() { gemm_warp_tiled<WT_BLOCK_M, WT_BLOCK_N, WT_K, WT_M_W, WT_N_W><<<grid_warp, WT_THREADS>>>(d_A, d_B, d_C, M, N, K); },
        d_C, bytes_C, C, C_ref, M, N, K
    ));


    // ========================================================
    // 6. BLOCK + SHARED + WARP + REGISTER TILING
    // ========================================================

    constexpr int RT_BLOCK_M = 64;
    constexpr int RT_BLOCK_N = 64;
    constexpr int RT_K = 16;

    constexpr int M_W = 32;
    constexpr int N_W = 32;

    constexpr int M_R = 8;
    constexpr int N_R = 4;

    constexpr int RT_WARPS = (RT_BLOCK_M / M_W) * (RT_BLOCK_N / N_W);
    constexpr int RT_THREADS = RT_WARPS * W;

    dim3 grid_register((N + RT_BLOCK_N - 1) / RT_BLOCK_N, (M + RT_BLOCK_M - 1) / RT_BLOCK_M);

    results.push_back(benchmark(
        "6. Block + shared + warp + register tiling",
        [&]() { gemm_register_tiled<RT_BLOCK_M, RT_BLOCK_N, RT_K, M_W, N_W, M_R, N_R><<<grid_register, RT_THREADS>>>(d_A, d_B, d_C, M, N, K); },
        d_C, bytes_C, C, C_ref, M, N, K
    ));


    // ========================================================
    // 7. TENSOR CORES
    // ========================================================

    half *d_A_half, *d_B_half;

    CUDA_CHECK(cudaMalloc(&d_A_half, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B_half, K * N * sizeof(half)));

    constexpr int convertThreads = 256;

    int blocks_A = (M * K + convertThreads - 1) / convertThreads;
    int blocks_B = (K * N + convertThreads - 1) / convertThreads;

    float_to_half<<<blocks_A, convertThreads>>>(d_A, d_A_half, M * K);
    float_to_half<<<blocks_B, convertThreads>>>(d_B, d_B_half, K * N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    constexpr int TC_BLOCK_M = 64;
    constexpr int TC_BLOCK_N = 64;
    constexpr int TC_K = 16;

    constexpr int TC_WARPS = (TC_BLOCK_M / WMMA_M) * (TC_BLOCK_N / WMMA_N);
    constexpr int TC_THREADS = TC_WARPS * W;

    dim3 grid_tensor((N + TC_BLOCK_N - 1) / TC_BLOCK_N, (M + TC_BLOCK_M - 1) / TC_BLOCK_M);

    results.push_back(benchmark(
        "7. Block + shared + warp + Tensor Cores",
        [&]() { gemm_tensor_core_shared<TC_BLOCK_M, TC_BLOCK_N, TC_K><<<grid_tensor, TC_THREADS>>>(d_A_half, d_B_half, d_C, M, N, K); },
        d_C, bytes_C, C, C_ref, M, N, K
    ));


    // ========================================================
    // cuBLAS SGEMM
    // ========================================================

    float alpha = 1.0f;
    float beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));

    results.push_back(benchmark(
        "10. cuBLAS SGEMM FP32",
        [&]() { CUBLAS_CHECK(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N)); },
        d_C, bytes_C, C, C_ref, M, N, K, 2, 2
    ));

    std::cout << "\n============================================================\n"
              << "LARGE MATRIX RESULTS\n"
              << "============================================================\n\n";

    print_header();

    for (size_t i = 0; i < results.size(); ++i) {
        double tolerance = (i == results.size() - 2) ? 5e-2 : 1e-2;
        print_result(results[i], tolerance);
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_C_ref));
    CUDA_CHECK(cudaFree(d_A_half));
    CUDA_CHECK(cudaFree(d_B_half));

    CUBLAS_CHECK(cublasDestroy(cublasHandle));

    return 0;
}

