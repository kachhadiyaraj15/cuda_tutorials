#include "kernels/7_kernel_resolve_bank_conflicts.cuh"
#include <cassert>

using uint = unsigned int;

namespace
{
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;

    template <const int BM_t, const int BN_t, const int BK_t, const int TM_t, const int TN_t>
    __global__ void sgemmResolveBankConflictsKernel(int M, int N, int K, float alpha,
                                              float *A, float *B, float beta,
                                              float *C) {
      const uint cRow = blockIdx.y;
      const uint cCol = blockIdx.x;

      // BN/TN are the number of threads to span a column
      const int threadCol = threadIdx.x % (BN_t / TN_t);
      const int threadRow = threadIdx.x / (BN_t / TN_t);

      // allocate space for the current blocktile in smem
      __shared__ float As[BM_t * BK_t];
      __shared__ float Bs[BK_t * BN_t];

      // Move blocktile to beginning of A's row and B's column
      A += cRow * BM_t * K;
      B += cCol * BN_t;
      C += cRow * BM_t * N + cCol * BN_t;

      // calculating the indices that this thread will load into SMEM
      // we'll load 128bit / 32bit = 4 elements per thread at each step
      const uint innerRowA = threadIdx.x / (BK_t / 4);
      const uint innerColA = threadIdx.x % (BK_t / 4);
      const uint innerRowB = threadIdx.x / (BN_t / 4);
      const uint innerColB = threadIdx.x % (BN_t / 4);

      // allocate thread-local cache for results in registerfile
      float threadResults[TM_t * TN_t] = {0.0};
      float regM[TM_t] = {0.0};
      float regN[TN_t] = {0.0};

      // outer-most loop over block tiles
      for (uint bkIdx = 0; bkIdx < K; bkIdx += BK_t) {
        // populate the SMEM caches
        // transpose A while loading it
        float4 tmp =
            reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM_t + innerRowA] = tmp.x;
        As[(innerColA * 4 + 1) * BM_t + innerRowA] = tmp.y;
        As[(innerColA * 4 + 2) * BM_t + innerRowA] = tmp.z;
        As[(innerColA * 4 + 3) * BM_t + innerRowA] = tmp.w;

        // "linearize" Bs while storing it
        tmp = reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
        Bs[((innerColB % 2) * 4 + innerRowB * 8 + 0) * 16 + innerColB / 2] = tmp.x;
        Bs[((innerColB % 2) * 4 + innerRowB * 8 + 1) * 16 + innerColB / 2] = tmp.y;
        Bs[((innerColB % 2) * 4 + innerRowB * 8 + 2) * 16 + innerColB / 2] = tmp.z;
        Bs[((innerColB % 2) * 4 + innerRowB * 8 + 3) * 16 + innerColB / 2] = tmp.w;
        __syncthreads();

        // advance blocktile
        A += BK_t;     // move BK columns to right
        B += BK_t * N; // move BK rows down

        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK_t; ++dotIdx) {
          // block into registers
          for (uint i = 0; i < TM_t; ++i) {
            regM[i] = As[dotIdx * BM_t + threadRow * TM_t + i];
          }
          for (uint i = 0; i < TN_t; ++i) {
            regN[i] = Bs[(dotIdx * 8 + i) * 16 + threadCol];
          }
          for (uint resIdxM = 0; resIdxM < TM_t; ++resIdxM) {
            for (uint resIdxN = 0; resIdxN < TN_t; ++resIdxN) {
              threadResults[resIdxM * TN_t + resIdxN] +=
                  regM[resIdxM] * regN[resIdxN];
            }
          }
        }
        __syncthreads();
      }

      // write out the results
      for (uint resIdxM = 0; resIdxM < TM_t; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN_t; resIdxN += 4) {
          // load C vector into registers
          float4 tmp = reinterpret_cast<float4 *>(
              &C[(threadRow * TM_t + resIdxM) * N + threadCol * TN_t + resIdxN])[0];
          // perform GEMM update in reg
          tmp.x = alpha * threadResults[resIdxM * TN_t + resIdxN] + beta * tmp.x;
          tmp.y = alpha * threadResults[resIdxM * TN_t + resIdxN + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[resIdxM * TN_t + resIdxN + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[resIdxM * TN_t + resIdxN + 3] + beta * tmp.w;
          // write back
          reinterpret_cast<float4 *>(
              &C[(threadRow * TM_t + resIdxM) * N + threadCol * TN_t + resIdxN])[0] =
              tmp;
        }
      }
    }
}

void sgemmResolveBankConflicts(
    const float *dA,
    const float *dB,
    float *dC,
    int M,
    int N,
    int K,
    cudaStream_t stream)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const dim3 block((BM * BN) / (TM * TN));
    const dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM);

    sgemmResolveBankConflictsKernel<BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(
        M, N, K, alpha, const_cast<float *>(dA), const_cast<float *>(dB), beta, dC);
}