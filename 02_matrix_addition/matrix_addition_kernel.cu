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
#define MATRIX_ROWS 4096
#define MATRIX_COLS 4096

__global__ void matrixAddKernel_1D(const float* A, const float* B, float* C, int rows, int cols){
    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIndex < (rows * cols)){
        C[globalIndex] = A[globalIndex] + B[globalIndex];
    }
}

// -- CUDA kernel for matrix addition using  2D thread Indexing --
__global__ void matrixAddKernel_2D(const float* A, const float* B, float* C, int rows, int cols){
    // Calculate the row and column index for the current thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = row * cols + col;

    // Check if the thread is within the bounds of the matrix
    if (row < rows && col < cols){
        C[idx] = A[idx] + B[idx];
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
    vector<float> h_C_gpu_1D(matrixSize); // Host Matrix C (GPU) 1D
    vector<float> h_C_gpu_2D(matrixSize); // Host Matrix C (GPU) 2D

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

    cout << "1D Config - Thread per block (1D): " << threadPerBlock_1D << endl;
    cout << "1D Config - Number of blocks (1D): " << numBlocks.x << endl;
    cout << "1D Config - Total CUDA threads launched: " << (long long)numBlocks.x * threadPerBlock_1D << endl;

    // -- Measure GPU execution time using CUDA events --
    cudaEvent_t start_gpu_event, stop_gpu_event;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_gpu_event));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_gpu_event));

    CHECK_CUDA_ERROR(cudaEventRecord(start_gpu_event, 0)); // Start recording the event

    // Launch the kernel with 1D configuration
    matrixAddKernel_1D<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);

    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors

    CHECK_CUDA_ERROR(cudaEventRecord(stop_gpu_event, 0)); // Stop recording the event
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_gpu_event));

    float gpu_time_ms_1D;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_time_ms_1D, start_gpu_event, stop_gpu_event)); // Calculate elapsed time

    cout<< "GPU Execution Time (Kernel only): " << gpu_time_ms_1D << " ms" << endl;

    long long total_ops_gpu = matrixSize; 
    double gpu_tflops_1D = calculateTFLOPS(total_ops_gpu, gpu_time_ms_1D);
    cout << "GPU Performance: " << gpu_tflops_1D << " TFLOPS/s" << endl;

    CHECK_CUDA_ERROR(cudaMemcpy(h_C_gpu_1D.data(), d_C, matrixSizeBytes, cudaMemcpyDeviceToHost)); // Copy result back to host
    CHECK_CUDA_ERROR(cudaEventDestroy(start_gpu_event));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_gpu_event));

    // -- GPU Matrix Addition (2D threds) --
    cout << "\n--- Performing GPU Matrix Addition (2D threads) ---\n" << endl;

    // define 2D thred block dimensions (16 x 16 )
    const int TILE_DIM = 16;
    dim3 threadPerBlock2D(TILE_DIM, TILE_DIM);

    // calculate 2D grid dimensions
    dim3 numBlocks2D((cols + threadPerBlock2D.x -1) / threadPerBlock2D.x,
                    (rows + threadPerBlock2D.y -1 )/ threadPerBlock2D.y);

    cout << "2D Config - Threads per block: " << threadPerBlock2D.x << "x" << threadPerBlock2D.y << endl;
    cout << "2D Config - Number of blocks: " << numBlocks2D.x << "x" << numBlocks2D.y << endl;
    cout << "2D Config - Total CUDA threads launched: " << (long long)numBlocks2D.x * numBlocks2D.y * threadPerBlock2D.x * threadPerBlock2D.y << endl;

    cudaEvent_t start_gpu_event_2D, stop_gpu_event_2D;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_gpu_event_2D));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_gpu_event_2D));

     // It's good practice to clear the output device buffer if you're reusing it,
    // although for simple addition, overwriting is fine.
    CHECK_CUDA_ERROR(cudaMemset(d_C, 0, matrixSizeBytes));

    CHECK_CUDA_ERROR(cudaEventRecord(start_gpu_event_2D, 0));
    matrixAddKernel_2D<<<numBlocks2D, threadPerBlock2D>>>(d_A, d_B, d_C, rows, cols);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaEventRecord(stop_gpu_event_2D, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_gpu_event_2D));

    float gpu_time_ms_2D;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_time_ms_2D, start_gpu_event_2D, stop_gpu_event_2D));
    std::cout << "GPU 2D Execution Time (Kernel only): " << gpu_time_ms_2D << " ms" << std::endl;
    double gpu_tflops_2D = calculateTFLOPS(total_ops_gpu, gpu_time_ms_2D);
    std::cout << "GPU 2D Performance: " << gpu_tflops_2D << " TFLOPS/s" << std::endl;

    CHECK_CUDA_ERROR(cudaMemcpy(h_C_gpu_2D.data(), d_C, matrixSizeBytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_gpu_event_2D));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_gpu_event_2D));

    // -- Verify the result --
    bool success_1D = true;
    for (size_t i = 0; i < matrixSize; i++){
        if (h_C_gpu_1D[i] != h_C_cpu[i]){
            cout << "Mismatch at index " << i << ": GPU = " << h_C_gpu_1D[i] << ", CPU = " << h_C_cpu[i] << endl;
            success_1D = false;
            break;
        }
    }
    if (success_1D){
        cout << "\nResults verified: CPU and GPU results match!" << endl;
    }
    else{
        cout << "\nResults verification failed!" << endl;
    }

    bool success_2D = true;
    for (size_t i = 0; i < matrixSize; ++i) {
        if (h_C_gpu_2D[i] != h_C_cpu[i]) {
            std::cerr << "2D GPU Mismatch at index " << i << ": CPU=" << h_C_cpu[i] << ", GPU=" << h_C_gpu_2D[i] << std::endl;
            success_2D = false;
            break;
        }
    }
    if (success_2D) {
        std::cout << "2D GPU results verified: Match CPU results." << std::endl;
    } else {
        std::cerr << "2D GPU results Mismatch!" << std::endl;
    }


    // -- Clean up --
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));

    // -- Log Results to results.txt --
    ofstream outFile("results.txt");
    if (outFile.is_open()){
        outFile << "Matrix Addition Performance Results (1D Threads vs 2D threads) \n";
        outFile << "---------------------------------------------\n";
        outFile << "Matrix Dimensions: " << rows << " x " << cols << "\n";
        outFile << "Total Elements: " << matrixSize << "\n";

        outFile << "CPU Performance:\n ";
        outFile << "Execution Time: " << cpu_time_ms << " ms\n";
        outFile << "Performance: " << cpu_tflops << " TFLOPS/s\n\n";
        
        outFile << "GPU 1D Thread Performance (Kernel only):\n";
        outFile << "  Execution Time: " << gpu_time_ms_1D << " ms\n";
        outFile << "  TFLOPS/s: " << gpu_tflops_1D << "\n\n";

        outFile << "GPU 2D Thread Performance (Kernel only):\n";
        outFile << "  Execution Time: " << gpu_time_ms_2D << " ms\n";
        outFile << "  TFLOPS/s: " << gpu_tflops_2D << "\n";
        outFile.close();
        cout << "\n Performance results logged to results.txt" << endl;
    }
    else{
        cout << "Unable to open results.txt for writing!" << endl;
    }
    return 0;

}
