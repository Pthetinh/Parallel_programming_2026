%%writefile test.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <iomanip>

using namespace std;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                 << cudaGetErrorString(error) << endl; \
            exit(1); \
        } \
    } while(0)

void fill_matrix(int* matrix, int N) {
    for(int i = 0; i < N * N; i++) {
        matrix[i] = rand() % 10;
    }
}

void write_matrix_to_file(string file_name, const int* matrix, int N) {
    ofstream file(file_name);
    if(!file.is_open()) {
        cerr << "Cannot open file: " << file_name << endl;
        return;
    }

    file << N << "\n";

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            file << matrix[i * N + j] << " ";
        }
        file << "\n";
    }

    file.close();
}

void multiply_cpu(int* A, int* B, int* C, int N) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            int sum = 0;
            for(int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

__global__ void matrix_mul_kernel(int* A, int* B, int* C, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(col < N && row < N) {
        int sum = 0;
        for(int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

bool compare_matrices(const int* A, const int* B, int N) {
    for (int i = 0; i < N * N; i++) {
        if (A[i] != B[i]) {
            return false;
        }
    }
    return true;
}

int main() {
    srand(42);

    vector<int> sizes = {200, 400, 800, 1200, 1600, 2000};

    vector<pair<int, int>> block_configs = {
        {8, 8},
        {16, 16},
        {32, 8},
        {32, 16}
    };

    ofstream timing_file("cuda_timing_results.csv");
    if (!timing_file.is_open()) {
        cerr << "Cannot open cuda_timing_results.csv" << endl;
        return 1;
    }

    timing_file << "MatrixSize,BlockX,BlockY,GridX,GridY,KernelTimeMs,Correct\n";

    for (int N : sizes) {
        cout << "\n========== Matrix size: " << N << " x " << N << " ==========\n";

        size_t bytes = N * N * sizeof(int);

        int* h_A = new int[N * N];
        int* h_B = new int[N * N];
        int* h_C = new int[N * N];
        int* h_C_cpu = new int[N * N];

        fill_matrix(h_A, N);
        fill_matrix(h_B, N);

        multiply_cpu(h_A, h_B, h_C_cpu, N);

        int *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

        for (auto cfg : block_configs) {
            int block_x = cfg.first;
            int block_y = cfg.second;

            dim3 block(block_x, block_y);
            dim3 grid(
                (N + block.x - 1) / block.x,
                (N + block.y - 1) / block.y
            );

            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));

            CUDA_CHECK(cudaMemset(d_C, 0, bytes));

            CUDA_CHECK(cudaEventRecord(start));

            matrix_mul_kernel<<<grid, block>>>(d_A, d_B, d_C, N);

            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaGetLastError());

            float milliseconds = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

            CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

            bool correct = compare_matrices(h_C, h_C_cpu, N);

            cout << "Block(" << block_x << "," << block_y << ") "
                 << "Grid(" << grid.x << "," << grid.y << ") "
                 << "Time = " << fixed << setprecision(3) << milliseconds << " ms, "
                 << "Correct = " << (correct ? "YES" : "NO") << endl;

            timing_file << N << ","
                        << block_x << ","
                        << block_y << ","
                        << grid.x << ","
                        << grid.y << ","
                        << milliseconds << ","
                        << (correct ? 1 : 0) << "\n";

            if (block_x == 16 && block_y == 16) {
                string out_name = "cuda_result_size_" + to_string(N) + "_block_" +
                                  to_string(block_x) + "x" + to_string(block_y) + ".txt";
                write_matrix_to_file(out_name, h_C, N);
            }

            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
        }

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));

        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        delete[] h_C_cpu;
    }

    timing_file.close();

    cout << "\nAll experiments finished. Results saved to cuda_timing_results.csv\n";
    return 0;
}