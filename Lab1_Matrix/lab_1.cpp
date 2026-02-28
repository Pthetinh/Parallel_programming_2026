#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <vector>

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

	vector<int> sizes = { 100, 200, 300, 400, 500, 1000 };

	ofstream time_file("timing_results.txt", ios::out | ios::trunc);

	time_file << "Size,Time_ms,FLOPs\n";

	for (int s = 0; s < sizes.size(); s++) {

		int size = sizes[s];

		cout << "==============================" << endl;
		cout << "Matrix size: " << size << " x " << size << endl;

		const string input_file = "matrix_size_" + to_string(size) + ".txt";
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

		auto start = std::chrono::steady_clock::now();

		multiply_non_omp(matrix1, matrix2, result, size);

		auto end = std::chrono::steady_clock::now();

		std::chrono::duration<double, std::milli> duration = end - start;

		const string output_file = "result_size_" + to_string(size) + ".txt";
		write_result_to_file(output_file, result, size);

		long long flops = 2LL * size * size * size;

		cout << "Time (milliseconds): " << duration.count() << endl;
		cout << "Task size (FLOPs): " << flops << endl;

		time_file << size << ","
			<< duration.count() << ","
			<< flops << "\n";

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