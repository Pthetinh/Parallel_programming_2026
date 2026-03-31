#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <string>

using namespace std;

void generate_matrix(const string& filename, int size) {
    ofstream file(filename);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            file << rand() % 10 << " ";
        }
        file << "\n";
    }
    file.close();
}

void read_file_to_matrix(const string& filename, int* matrix, int size) {
    ifstream file(filename);
    for (int i = 0; i < size * size; i++) {
        file >> matrix[i];
    }
    file.close();
}

void write_result_to_file(const string& filename, int* matrix, int size) {
    ofstream file(filename);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            file << matrix[i * size + j] << " ";
        }
        file << "\n";
    }
    file.close();
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, process_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    vector<int> sizes = {200, 400, 800, 1200, 1600, 2000};

    ofstream timing_file;
    if (rank == 0) {
        timing_file.open("timing_results.txt");
    }

    for (int s = 0; s < sizes.size(); s++) {

        int N = sizes[s];

        int* A = nullptr;
        int* B = nullptr;
        int* C = nullptr;

        if (rank == 0) {
            A = new int[N * N];
            B = new int[N * N];
            C = new int[N * N];


            const string input_file_A = "matrixA_size_" + to_string(N) + ".txt";
            const string input_file_B = "matrixB_size_" + to_string(N) + ".txt";
            generate_matrix(input_file_A, N);
            generate_matrix(input_file_B, N);

            read_file_to_matrix(input_file_A, A, N);
            read_file_to_matrix(input_file_B, B, N);
        } else {
            B = new int[N * N];
        }

        MPI_Bcast(B, N * N, MPI_INT, 0, MPI_COMM_WORLD);


        int* sendcounts = new int[process_count];
        int* displs = new int[process_count];

        int base = N / process_count;
        int remainder = N % process_count;

        int offset = 0;
        for (int i = 0; i < process_count; i++) {
            int rows = base + (i < remainder ? 1 : 0);
            sendcounts[i] = rows * N;
            displs[i] = offset;
            offset += rows * N;
        }

        int local_size = sendcounts[rank];
        int local_rows = local_size / N;

        int* local_A = new int[local_size];
        int* local_C = new int[local_size];

        MPI_Scatterv(A, sendcounts, displs, MPI_INT,
                     local_A, local_size, MPI_INT,
                     0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();


        for (int i = 0; i < local_rows; i++) {
            for (int j = 0; j < N; j++) {
                local_C[i * N + j] = 0;
                for (int k = 0; k < N; k++) {
                    local_C[i * N + j] += local_A[i * N + k] * B[k * N + j];
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double end = MPI_Wtime();

        MPI_Gatherv(local_C, local_size, MPI_INT,
                    C, sendcounts, displs, MPI_INT,
                    0, MPI_COMM_WORLD);


        if (rank == 0) {
            double time = end - start;

            cout << "Size: " << N
                 << ", Processes: " << process_count
                 << ", Time: " << time << " s\n";

            timing_file << N << " "
                        << process_count << " "
                        << time << "\n";
            
            const string output_file = "result_size_" + to_string(N) + ".txt";
            write_result_to_file(output_file, C, N);
        }


        delete[] local_A;
        delete[] local_C;
        delete[] sendcounts;
        delete[] displs;

        if (rank == 0) {
            delete[] A;
            delete[] B;
            delete[] C;
        } else {
            delete[] B;
        }
    }

    if (rank == 0) {
        timing_file.close();
    }

    MPI_Finalize();
    return 0;
}
