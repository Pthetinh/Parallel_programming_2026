#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <vector>
#include <omp.h>

using namespace std;

template<class T>
void multiply_non_omp(T** matrix1, T** matrix2, T** result, int size) {
	for (int row = 0; row < size; row++) {
		for (int col = 0; col < size; col++) {
			T sum = 0;
			for (int i = 0; i < size; i++) {
				sum += matrix1[row][i] * matrix2[i][col];
			}
			result[row][col] = sum;
		}
	}
}


template<class T>
void multiply_omp(T** matrix1, T** matrix2, T** result, int size, int num_thread) {
#pragma omp parallel for num_threads(num_thread)
	for (int row = 0; row < size; row++) {
		for (int col = 0; col < size; col++) {
			T sum = 0;
			for (int i = 0; i < size; i++) {
				sum += matrix1[row][i] * matrix2[i][col];
			}
			result[row][col] = sum;
		}
	}
}


void generate_matrix(const string& file_path, int size) {
	ofstream data(file_path);

	if (!data.is_open()) {
		std::cerr << "Cannot open file\n";
		return;
	}

	data << size << "\n";

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			data << (rand() % size + 10) << " ";
		}
		data << endl;
	}

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			data << (rand() % size - 15) << " ";
		}
		data << endl;
	}
	data.close();
}

template<class T>
void read_file_to_matrix(const string& file_path, T** matrix1, T** matrix2, int& size) {
	ifstream file(file_path);

	if (!file.is_open()) {
		cerr << "Cannot open file: " << file_path << endl;
		return;
	}

	file >> size;

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			file >> matrix1[i][j];
		}
	}

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			file >> matrix2[i][j];
		}
	}

	file.close();
}

template<class T>
void write_result_to_file(const string& file_path, T** result, int size) {
	ofstream out_put(file_path);

	out_put << size << endl;

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			out_put << result[i][j] << " ";
		}
		out_put << endl;
	}

	out_put.close();
}

int main() {

	vector<int> sizes = { 200, 400, 800, 1200, 1600, 2000 };
	vector<int> thread_counts = {1, 2, 4, 8 };

	ofstream time_file("timing_results_all.csv", ios::out | ios::trunc);
	time_file << "Mode,Threads,Size,Time_ms,FLOPs\n";

	for (int s = 0; s < sizes.size(); s++) {
		int size = sizes[s];

		cout << "========================================\n";
		cout << "Matrix size: " << size << " x " << size << endl;

		string input_file = "matrix_size_" + to_string(size) + ".txt";
		generate_matrix(input_file, size);

		int** matrix1 = new int* [size];
		int** matrix2 = new int* [size];
		int** result = new int* [size];

		for (int i = 0; i < size; i++) {
			matrix1[i] = new int[size];
			matrix2[i] = new int[size];
			result[i] = new int[size];
		}

		read_file_to_matrix(input_file, matrix1, matrix2, size);

		long long flops = 2LL * size * size * size;

		// ===== Non-OMP =====
		//auto start_non = chrono::steady_clock::now();
		//multiply_non_omp(matrix1, matrix2, result, size);
		//auto end_non = chrono::steady_clock::now();

		//chrono::duration<double, milli> duration_non = end_non - start_non;

		//cout << "Non-OMP"
		//	<< " | Time (ms): " << duration_non.count()
		//	<< " | FLOPs: " << flops << endl;

		//time_file << "Non-OMP,1,"
		//	<< size << ","
		//	<< duration_non.count() << ","
		//	<< flops << "\n";

		//write_result_to_file("result_size_" + to_string(size) + "_non_omp.txt", result, size);

		// ===== OMP =====
		for (int t = 0; t < thread_counts.size(); t++) {

			int num_threads = thread_counts[t];

			auto start_omp = chrono::steady_clock::now();
			multiply_omp(matrix1, matrix2, result, size, num_threads);
			auto end_omp = chrono::steady_clock::now();

			chrono::duration<double, milli> duration_omp = end_omp - start_omp;

			cout << "OMP"
				<< " | Threads: " << num_threads
				<< " | Time (ms): " << duration_omp.count()
				<< " | FLOPs: " << flops << endl;

			time_file << "OMP,"
				<< num_threads << ","
				<< size << ","
				<< duration_omp.count() << ","
				<< flops << "\n";

			string output_file = "result_size_" + to_string(size) +
				"_threads_" + to_string(num_threads) + ".txt";
			write_result_to_file(output_file, result, size);
		}

		for (int i = 0; i < size; i++) {
			delete[] matrix1[i];
			delete[] matrix2[i];
			delete[] result[i];
		}

		delete[] matrix1;
		delete[] matrix2;
		delete[] result;
	}

	time_file.close();
	return 0;
}