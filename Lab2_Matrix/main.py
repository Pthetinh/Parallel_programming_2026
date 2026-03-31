import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_file_to_matrix(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        size = int(f.readline().strip())
        data = np.loadtxt(f)

    if data.shape != (2 * size, size):
        raise ValueError(f"{file_path}: expected {(2*size, size)} but got {data.shape}")

    matrix1 = data[:size, :]
    matrix2 = data[size:, :]
    return size, matrix1, matrix2


def read_result_matrix(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        size = int(f.readline().strip())
        data = np.loadtxt(f)

    if data.shape != (size, size):
        raise ValueError(f"{file_path}: expected {(size, size)} but got {data.shape}")

    return size, data


def multiply_matrices(matrix1, matrix2):
    return np.dot(matrix1, matrix2)


def check_one(size: int, thread: int, base_dir=".", base_dir_result_mul="."):
    input_file = f"{base_dir}/matrix_size_{size}.txt"
    result_file = f"{base_dir_result_mul}/result_size_{size}_threads_{thread}.txt"

    in_size, A, B = read_file_to_matrix(input_file)
    out_size, C_cpp = read_result_matrix(result_file)

    if in_size != size or out_size != size:
        raise ValueError(f"Size mismatch. expected {size}, got input={in_size}, output={out_size}")

    C_np = multiply_matrices(A, B)

    return np.array_equal(C_cpp.astype(int), C_np.astype(int))


def main():
    sizes = [200, 400, 800, 1200, 1600, 2000]
    threads = [1, 2, 4, 8]

    for t in threads:
        all_ok = True
        for s in sizes:
            try:
                ok = check_one(s, t,  base_dir="Data_Matrix", base_dir_result_mul="Data_mul_matrix")
                print(f"Thread {t} - Size {s}: {'PASS' if ok else 'FAIL'}")
                all_ok = all_ok and ok
            except Exception as e:    
                print(f"Thread {t} - Size {s}: ERROR -> {e}")
                all_ok = False
    
    if all_ok:
        print("🔥Congratulation, all tests were passed!")
    else:
        print("❌ Some tests failed!🔥")

if __name__ == "__main__":
    main()