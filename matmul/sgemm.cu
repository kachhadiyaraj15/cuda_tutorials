#include<cuda_runtime.h>
#include<cublas_v2.h>

#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<fstream>
#include<iostream>
#include<string>
#include<vector>

using namespace std;

#define CUDA_CHECK(call) checkCuda((call), __FILE__, __LINE__)
#define CUBLAS_CHECK(call) checkCublas((call), __FILE__, __LINE__)

void checkCuda(cudaError_t err, const char* file, int line){
    if (err != cudaSuccess){
        cerr << "CUDA Error at" << file << ":" << line << "->" << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

void checkCublas(cublasStatus_t status, const char* file, int line){
    if (status != CUBLAS_STATUS_SUCCESS){
        cerr << "cuBLAS error at " << file << ":" << line << " -> status code " << static_cast<int>(status)<<endl;
        exit(EXIT_FAILURE);
    }
}

struct GemmShape{
    int M;
    int N;
    int K;
};

vector<GemmShape> getTestShapes(){
    return {
        {128, 128, 128},
        {256, 256, 256},
        {512, 256, 128},
        {1024, 512, 256},
        {512, 1024, 512},
        {1024, 1024, 1024},
        {2048, 1024, 512},
        {2048, 2048,2048},
        {4096, 512, 4096},
        {4096, 1024, 2048},
        {4096, 2048, 4096},
        {4096, 4096, 4096}
    };
}

void randomizeMatrix(float* mat, int count){
    for (int i=0; i<count; i++){
        float value=static_cast<float>(std::rand() % 5) + 0.01f * static_cast<float>(std::rand()%5);
        if (rand() % 2 ==0){
            value = -value;
        }
        mat[i] = value;
    }
}

void zeroMatrix(float* mat, int count){
    for (int i=0; i<count; i++){
        mat[i] = 0.0f;
    }
}

double computeGflops(int M, int N, int K, double seconds){
    double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N)* static_cast<double>(K);
    return (flops * 1e-9)/seconds;
}

void appendCsvRow(const string& path,
const string& impl,
int M, int N, int K,
double avg_ms,
double gflops){
    bool write_header = false;
    ifstream in (path);
    if (!in.good()){
        write_header=true;
    }
    in.close();

    ofstream out(path, ios::app);
    if (!out.is_open()){
        cerr<< "Failed to open result file: " <<path <<endl;
        exit(EXIT_FAILURE);
    }

    if(write_header){
        out << "impl,M,N,K,avg_ms,gflops\n";
    }

    out << impl << ","
        << M << ","
        << N << ","
        << K << ","
        << avg_ms << ","
        << gflops << "\n";
}

void runCublasFP32(cublasHandle_t handle, 
int M, int N, int K,
float alpha,
const float* dA, const float* dB,
float beta,
float* dC){
    // Row major A and B are handled by swapping A/B in the buBLAS call
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, M, K,
        &alpha,
        dB, CUDA_R_32F, N,
        dA, CUDA_R_32F, K,
        &beta,
        dC, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}


int main(){
    srand(0);

    const string result_file="benchmark_results/cublass_results.csv";
    const int repeat = 20;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUDA_CHECK(cudaSetDevice(0));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    vector<GemmShape> shapes = getTestShapes();

    for (const auto& shape: shapes){
        int M = shape.M;
        int N = shape.N;
        int K = shape.K;

        size_t bytesA = static_cast<size_t>(M) * K * sizeof(float);
        size_t bytesB = static_cast<size_t>(K) * N * sizeof(float);
        size_t bytesC = static_cast<size_t>(M)* N * sizeof(float);

        float* hA = static_cast<float*>(malloc(bytesA));
        float* hB = static_cast<float*>(malloc(bytesB));
        float* hC = static_cast<float*>(malloc(bytesC));

        if (!hA || !hB || !hC) {
            std::cerr << "Host allocation failed" << std::endl;
            return EXIT_FAILURE;
        }

        float* dA = nullptr;
        float* dB = nullptr;
        float* dC = nullptr;

        CUDA_CHECK(cudaMalloc(&dA, bytesA));
        CUDA_CHECK(cudaMalloc(&dB, bytesB));
        CUDA_CHECK(cudaMalloc(&dC, bytesC));

        CUDA_CHECK(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dC, hC, bytesC, cudaMemcpyHostToDevice));

        // Warm-up
        runCublasFP32(handle, M, N, K, alpha, dA, dB, beta, dC);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for(int i=0; i<repeat; i++){
            runCublasFP32(handle, M,N, K, alpha, dA, dB, beta, dC);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

        double avg_ms = static_cast<double>(total_ms) / static_cast<double>(repeat);
        double avg_seconds = avg_ms / 1000.0;
        double gflops = computeGflops(M, N,K, avg_seconds);

        appendCsvRow(result_file, "cublas", M, N, K, avg_ms, gflops);

        CUDA_CHECK(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));
        
        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));

        free(hA);
        free(hB);
        free(hC);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));

    return 0;
}
