#include "kernels/6_kernel_vectorize.cuh"
#include <cassert>

namespace{
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;

    __global__ void sgemm_vectorize(
        int M, int N, int K, 
        float alpha, 
        const float* A,
        const float* B,
        float beta,
        float* C
    ){
        // block row and column 
        const int cRow = blockIdx.y;
        const int cCol = blockIdx.x;

        // BN/TN are the number of threads to span a column
        const int threadCol = threadIdx.x / (BN / TN);
        const int threadRow = threadIdx.y % (BN/TN);

        // allocate space for the current blocktile in smem
        __shared__ float As[BM * BK];
        __shared__ float Bs[BK * BN];

        // move blocktile to beginning of A's and B's column
        A += cRow * BM * K;
        B += cCol * BN;
        C += cRow * BM * N + cCol * BN;

        // calculating the indices that his thread will load into smem, we'll load 128bit/32bit = 4 elements per thread at each steup 
        const int innerRowA = threadIdx.x / (BK/4);
        const int innerColA = threadIdx.x % (BK/4);
        const int innerRowB = threadIdx.x / (BN/4);
        const int innerColB = threadIdx.x % (BN/4);

        // allocate thread-local cache for results in registerfile
        float threadResults[TM * TN] = {0.0};
        float regM[TM] = {0.0};
        float regN[TN] = {0.0};

        // outer-nost loop over block tiles
        for(int bkIdx = 0; bkIdx < K; bkIdx += BK){
            // populate the SMEM caches, transpose A while laoding it 
            float4 tmp = reinterpret_cast<const float4 *>(&A[innerRowA * K + innerColA * 4])[0];
            As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;
            
            reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] = reinterpret_cast<const float4 *>(&B[innerRowB * N + innerColB * 4])[0];
            __syncthreads();

            // advance blocktile
            A += BK; // move BK columns to right
            B += BK * N; // move BK rows then

            // calculate per-thread results
            for(int dotIdx = 0; dotIdx < BK; ++dotIdx){
                // block into registers
                for(int i = 0; i < TM; i++){
                    regM[i] = As[dotIdx * BM + threadRow * TM + i];
                }
                for(int i=0; i< TN; ++i){
                    regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
                }
                for(int resIdxM = 0; resIdxM < TM; ++resIdxM){
                    for(int resIdxN = 0; resIdxN < TN; ++resIdxN){
                        threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
                    }
                }
            }
            __syncthreads();
        }

        // write out the results
        for(int resIdxM = 0; resIdxM < TM; resIdxM += 1){
            for(int resIdxN = 0; resIdxN < TN; resIdxN += 4){
                // load C vector int oregisters
                float4 tmp = reinterpret_cast<float4 *>(&C[(threadRow * TM + resIdxM)* N+ threadCol * TN + resIdxN])[0];
                // perform GEMM update in reg
                tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
                tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
                tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
                tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;

                // write back
                reinterpret_cast<float4 *>(&C[(threadRow *TM + resIdxM) * N + threadCol * TN + resIdxN])[0] = tmp;
            }
        }
    }

}

void sgemmVectorize(
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
    const dim3 block((BM * BN)/ (TM * TN));
    const dim3 grid(
        ((N + BN -1) / BN),
        ((M + BM -1) / BM)
    );

    sgemm_vectorize<<<grid, block, 0, stream>>>(M, N, K, alpha, dA, dB, beta, dC);
}