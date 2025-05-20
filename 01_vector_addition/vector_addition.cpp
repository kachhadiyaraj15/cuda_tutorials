#include <iostream>  // console printing 
#include <fstream>  // writing to file
#include <vector>   // dynamic arrays 
#include <chrono> // time measurement

// `extern` is used to declare that the function is defined elsewhere
extern void run_gpu_add(const int *h_a, const int *h_b, int *h_c, int N, int warmups, int repeats, double &avg_time_ms); 

void cpu_add(const int *a, const int *b, int *c, int N){
    for (int i = 0; i < N; i++)
    {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    const int N = 1 << 20; // 1M elements 
    const int WARMUPS = 3;
    const int REPEATS = 10;

    std::vector<int> h_a(N), h_b(N), h_c_cpu(N), h_c_gpu(N);
    for (int i=0; i < N; i++){
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    // CPU timing 
    double total_cpu_time = 0;
    for (int i =0 ; i< WARMUPS; i++)
    {
        cpu_add(h_a.data(), h_b.data(), h_c_cpu.data(), N);
    }

    for (int i=0; i< REPEATS; i++)
    {
        auto start = std :: chrono::high_resolution_clock::now();
        cpu_add(h_a.data(), h_b.data(), h_c_cpu.data(), N);
        auto end = std :: chrono::high_resolution_clock::now();
        total_cpu_time += std::chrono::duration<double, std::milli>(end - start).count();
    }

    double avg_cpu_time = total_cpu_time / REPEATS;

    // GPU timing
    double avg_gpu_time = 0;
    run_gpu_add(h_a.data(), h_b.data(), h_c_gpu.data(), N, WARMUPS, REPEATS, avg_gpu_time);

    // Result Verification 
    bool match = true;
    for (int i=0; i <N ; i++){
        if (h_c_cpu[i] != h_c_gpu[i]){
            match = false;
            std::cerr << "Mismatch at index " << i << "\n";
            break;
        }
    }

    // Write results
    std::ofstream result("results.txt");
    result << "N: " << N << "\n";
    result << "CPU avg time (ms): " << avg_cpu_time << "\n";
    result << "GPU avg time (ms): " << avg_gpu_time << "\n";
    result << "Results match: " << (match ? "Yes" : "No") << "\n";
    result.close();

    std::cout << "Results written to results.txt\n";
    return 0;
}