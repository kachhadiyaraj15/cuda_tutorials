#include "kernels/4_kernel_1D_blocktiling.cuh"
#include <cassert>

namespace{

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 8;
constexpr int TM = 8;

template <const int BLOCKSIZE>

__global__ void sgemm_1d_blocktiling(
    int M, int N, int K, 
    float alpha, 
    const float* A,
    const float* B,
    float beta,
    float* C
){
    // we are considering that rows of A and columns of B as a matrix calcualaion 
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    // each warp will calculate 32*TM elements, with 32 being the columnar dim 
    // TM means vertical number of elements calaulate by one thread
    const int threadCol = threadIdx.x % BN;
    const int threadRow = threadIdx.x / BN;

    // allocate space for the current blocktile in SMEM
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // move blocktile to beginnning of A's row and B's column 
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;
    
    // assert(BM * BK == blockDim.x);
    // assert(BN * BK == blockDim.x);
    const int innerColA = threadIdx.x % BK;
    const int innerRowA = threadIdx.x / BK;
    const int innerColB = threadIdx.x % BN;
    const int innerRowB = threadIdx.x / BN;

    // allocate thread-local cache for results in registerfile
    float threadResults[TM] = {0.0};

    // outer loop over block titles
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK){
        // populate the SMEM caches 
        // if the coordinates are safe (less than M and K), we load from A, if not, we load 0.0f
        int globalRowA = cRow * BM + innerRowA;
        int globalColA = bkIdx + innerColA;
        As[innerRowA * BK + innerColA] = ( globalRowA< M && globalColA<K ? A[innerRowA * K + innerColA] : 0.0f);

        int globalRowB = bkIdx + innerRowB;
        int globalColB = cCol * BN + innerColB;
        Bs[innerRowB * BN + innerColB] = (globalRowB<K && globalColB<N ? B[innerRowB * N + innerColB] : 0.0f);

        __syncthreads();

        //advance blocktile
        A += BK;
        B += BK * N;

        // calculate per-thread results 
        for (int dotIdx = 0; dotIdx <BK; ++dotIdx){
            // we make the dotproducts loop the outside loop, which facilitates reuse of the Bs entery, which we can cache in a tmp var
            float tmpB = Bs[dotIdx * BN + threadCol];
            for (int resIdx = 0; resIdx < TM; ++resIdx){
                threadResults[resIdx] += As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
            }
        }
        __syncthreads();
    }
    
    // write out the results 
    for (int resIdx = 0; resIdx < TM ; ++resIdx){
        int globalRowC = cRow * BM + threadRow * TM + resIdx;
        int globalColC = cCol * BN + threadCol;
        if (globalRowC < M && globalColC < N){
            C[(threadRow * TM + resIdx) * N + threadCol] = alpha * threadResults[resIdx] + beta * C[(threadRow * TM + resIdx) * N + threadCol];
        }
    }
}

}


void sgemm1DBlocktiling(
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
    const dim3 block((BM * BN)/TM);
    const dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );

    sgemm_1d_blocktiling<kBlockSize><<<grid, block, 0, stream>>>(M, N, K, alpha, dA, dB, beta, dC);
}

