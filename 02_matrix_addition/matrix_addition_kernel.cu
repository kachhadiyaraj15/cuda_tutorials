#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <cmath>

using namespace std;

// -- CUDA Error Handling Macro --
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

// helper function to check CUDA errors
void check(cudaError_t err, const char* const func, const char* const file, const int line){
    if (err != cudaSuccess) {
        cerr << "CUDA error at " << file << ":" << line << ends;
        cerr << "Function:" << func << endl;
        cerr << "Error: " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE); // Exit with a failure status
    }
}

// Define constants for matrix dimensions (you can change these)
#define MATRIX_ROWS 2048
#define MATRIX_COLS 2048

__global__ void matrixAddKernel(const float* A, const float* B, float* C, int rows, int cols){
    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIndex < (rows * cols)){
        C[globalIndex] = A[globalIndex] + B[globalIndex];
    }
}

// Function to perform matrix addition on the CPU (for comparison)
void matrixAddCPU(const vector<float>& A, const vector<float>& B, vector<float>& C, int rows, int cols) {
    for (int i=0; i<rows; i++){
        for (int j = 0; j < cols; j++){
            C[i * cols + j] = A[i * cols + j] + B[i * cols + j];
        }
    }
}

// Function to calculate GFLOPS/TFLOPS
// TFLOPS/s = (Total Floating-Point Operations) / (Execution Time in Seconds * 10^12)
double calculateTFLOPS(long long total_ops, double execution_time_ms){
    if (execution_time_ms == 0) return 0.0;
    double execution_time_sec = execution_time_ms / 1000.0;
    return (static_cast<double>(total_ops) / execution_time_sec) / 1e12 ;
}

// Main Function 
int main(){
    // Matrix Dimentions
    const int rows = MATRIX_ROWS;
    const int cols = MATRIX_COLS;
    const size_t matrixSize = static_cast<size_t>(rows) * cols;
    const size_t matrixSizeBytes = matrixSize * sizeof(float);

    cout << "Matrix Dimensions: " << rows << " x " << cols << endl;
    cout << "Total Elements: " << matrixSize << endl;

    vector<float> h_A(matrixSize); // Host Matrix A
    vector<float> h_B(matrixSize); // Host Matrix B
    vector<float> h_C_cpu(matrixSize); // Host Matrix C (CPU)
    vector<float> h_C_gpu(matrixSize); // Host Matrix C (GPU)

    // Initialize host matrices A and B with some values
    for (size_t i = 0 ; i< matrixSize; i++){
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // --- CPU Matrix Addition ---
    cout << "\nPerforming CPU matrix addition...." << endl;
    auto start_cpu = chrono::high_resolution_clock::now();
    matrixAddCPU(h_A, h_B, h_C_cpu, rows, cols);
    auto end_cpu = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> cpu_duration = end_cpu - start_cpu;
    double cpu_time_ms = cpu_duration.count();
    cout << "CPU Execution Time: " << cpu_time_ms << " ms" << endl;

    long long total_ops_cpu = matrixSize; // Each element involves 1 addition operation
    double cpu_tflops = calculateTFLOPS(total_ops_cpu, cpu_time_ms);
    cout << "CPU Performance: " << cpu_tflops << " TFLOPS/s" << endl;

    // --- GPU Matrix Addition ---
    cout << "\nPerforming GPU matrix addition...." << endl;

    // Device (GPU) memory pointers
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, matrixSizeBytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, matrixSizeBytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, matrixSizeBytes));

    // Copy matrices A and B from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A.data(), matrixSizeBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B.data(), matrixSizeBytes, cudaMemcpyHostToDevice));

    // Define 1D thread bock dimensions
    // A typical block size for 1D problems is 256, 512, or 1024 threads.
    // 256 is a good general choice that fills SMa effectively.
    const int threadPerBlock_1D = 256;
    dim3 threadsPerBlock(threadPerBlock_1D); // only x-dimension is used for 1D kernel

    // Calculate 1D grid dimensions
    dim3 numBlocks((matrixSize + threadPerBlock_1D - 1) / threadPerBlock_1D); 

    cout << "Thread per block (1D): " << threadPerBlock_1D << endl;
    cout << "Number of blocks (1D): " << numBlocks.x << endl;
    cout << "Total CUDA threads launched: " << (long long)numBlocks.x * threadPerBlock_1D << endl;

    // -- Measure GPU execution time using CUDA events --
    cudaEvent_t start_gpu_event, stop_gpu_event;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_gpu_event));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_gpu_event));

    CHECK_CUDA_ERROR(cudaEventRecord(start_gpu_event, 0)); // Start recording the event

    // Launch the kernel with 1D configuration
    matrixAddKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);

    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors

    CHECK_CUDA_ERROR(cudaEventRecord(stop_gpu_event, 0)); // Stop recording the event
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_gpu_event));

    float gpu_time_ms;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_time_ms, start_gpu_event, stop_gpu_event)); // Calculate elapsed time

    cout<< "GPU Execution Time (Kernel only): " << gpu_time_ms << " ms" << endl;

    long long total_ops_gpu = matrixSize; 
    double gpu_tflops = calculateTFLOPS(total_ops_gpu, gpu_time_ms);
    cout << "GPU Performance: " << gpu_tflops << " TFLOPS/s" << endl;

    CHECK_CUDA_ERROR(cudaMemcpy(h_C_gpu.data(), d_C, matrixSizeBytes, cudaMemcpyDeviceToHost)); // Copy result back to host

    // -- Verify the result --
    bool success = true;
    for (size_t i = 0; i < matrixSize; i++){
        if (h_C_gpu[i] != h_C_cpu[i]){
            cout << "Mismatch at index " << i << ": GPU = " << h_C_gpu[i] << ", CPU = " << h_C_cpu[i] << endl;
            success = false;
            break;
        }
    }
    if (success){
        cout << "\nResults verified: CPU and GPU results match!" << endl;
    }
    else{
        cout << "\nResults verification failed!" << endl;
    }

    // -- Clean up --
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_gpu_event));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_gpu_event));

    // -- Log Results to results.txt --
    ofstream outFile("results.txt");
    if (outFile.is_open()){
        outFile << "Matrix Addition Performance Results (1D Threads) \n";
        outFile << "---------------------------------------------\n";
        outFile << "Matrix Dimensions: " << rows << " x " << cols << "\n";
        outFile << "Total Elements: " << matrixSize << "\n";

        outFile << "CPU Performance:\n ";
        outFile << "Execution Time: " << cpu_time_ms << " ms\n";
        outFile << "Performance: " << cpu_tflops << " TFLOPS/s\n\n";
        
        outFile << "GPU Performance (Kernel Only):\n ";
        outFile << "Execution Time: " << gpu_time_ms << " ms\n";
        outFile << "Performance: " << gpu_tflops << " TFLOPS/s\n\n";
        outFile.close();
        cout << "\n Performance results logged to results.txt" << endl;
    }
    else{
        cout << "Unable to open results.txt for writing!" << endl;
    }
    return 0;

}
