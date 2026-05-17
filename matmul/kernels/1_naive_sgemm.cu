#include "kernels/1_naive_sgemm.cuh"

namespace {

constexpr int kTile = 16;

__global__ void naiveSgemmKernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}

}  // namespace

void launchNaiveSgemm(
    const float* dA,
    const float* dB,
    float* dC,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
    const dim3 block(kTile, kTile);
    const dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    naiveSgemmKernel<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
}

