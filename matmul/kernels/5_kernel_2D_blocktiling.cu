#include "kernels/5_kernel_2D_blocktiling.cuh"
#include <cassert>

namespace{

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;

__global__ void sgemm_2d_blocktiling(
    int M, int N, int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C
){
    // Block row and column
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    // Thread coordinates inside the thread block
    // Thread block size is (BM * BN) / (TM * TN) = 128 * 128 / 64 = 256 threads.
    // The thread block is arranged as a 2D grid of threads: (BM/TM) x (BN/TN) = 16 x 16 threads.
    const int threadCol = threadIdx.x % (BN / TN); // 0 to 15
    const int threadRow = threadIdx.x / (BN / TN); // 0 to 15

    // Allocate space for the current block tile in shared memory (SMEM)
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Move pointers to the starting positions of the block tile
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // Registers to store the intermediate dot-product accumulations
    float threadResults[TM][TN] = {0.0f};

    // Registers to cache values loaded from shared memory
    float regA[TM];
    float regB[TN];

    // Loop over the K-dimension tiles
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Load A tile from global memory to shared memory
        // As has size BM * BK = 128 * 8 = 1024 floats.
        // With 256 threads, each thread loads 4 elements.
        for (int loadOffset = 0; loadOffset < BM * BK; loadOffset += 256) {
            int loadIdx = threadIdx.x + loadOffset;
            int rowA = loadIdx / BK;
            int colA = loadIdx % BK;
            int globalRowA = cRow * BM + rowA;
            int globalColA = bkIdx + colA;
            if (globalRowA < M && globalColA < K) {
                As[rowA * BK + colA] = A[rowA * K + colA];
            } else {
                As[rowA * BK + colA] = 0.0f;
            }
        }

        // Load B tile from global memory to shared memory
        // Bs has size BK * BN = 8 * 128 = 1024 floats.
        // With 256 threads, each thread loads 4 elements.
        for (int loadOffset = 0; loadOffset < BK * BN; loadOffset += 256) {
            int loadIdx = threadIdx.x + loadOffset;
            int rowB = loadIdx / BN;
            int colB = loadIdx % BN;
            int globalRowB = bkIdx + rowB;
            int globalColB = cCol * BN + colB;
            if (globalRowB < K && globalColB < N) {
                Bs[rowB * BN + colB] = B[rowB * N + colB];
            } else {
                Bs[rowB * BN + colB] = 0.0f;
            }
        }

        // Synchronize to ensure all elements are loaded into shared memory
        __syncthreads();

        // Advance pointers for next iteration
        A += BK;
        B += BK * N;

        // Perform the dot product computation on the current block tile
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Cache values from shared memory to registers
            for (int i = 0; i < TM; ++i) {
                regA[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }
            for (int j = 0; j < TN; ++j) {
                regB[j] = Bs[dotIdx * BN + threadCol * TN + j];
            }

            // Perform outer product calculation using registers
            for (int resRow = 0; resRow < TM; ++resRow) {
                for (int resCol = 0; resCol < TN; ++resCol) {
                    threadResults[resRow][resCol] += regA[resRow] * regB[resCol];
                }
            }
        }

        // Synchronize before next shared memory load
        __syncthreads();
    }

    // Write the results back to global memory C
    for (int resRow = 0; resRow < TM; ++resRow) {
        for (int resCol = 0; resCol < TN; ++resCol) {
            int globalRowC = cRow * BM + threadRow * TM + resRow;
            int globalColC = cCol * BN + threadCol * TN + resCol;
            if (globalRowC < M && globalColC < N) {
                int offset = (threadRow * TM + resRow) * N + (threadCol * TN + resCol);
                C[offset] = alpha * threadResults[resRow][resCol] + beta * C[offset];
            }
        }
    }
}

} // namespace

void sgemm2DBlocktiling(
    const float* dA,
    const float* dB,
    float* dC,
    int M,
    int N,
    int K,
    cudaStream_t stream
){
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const dim3 block((BM * BN) / (TM * TN));
    const dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );

    sgemm_2d_blocktiling<<<grid, block, 0, stream>>>(M, N, K, alpha, dA, dB, beta, dC);
}