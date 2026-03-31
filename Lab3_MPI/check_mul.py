import numpy as np

def read_file_to_matrix(file1_path: str, file2_path: str, size: int):
    with open(file1_path, "r", encoding="utf-8") as f1:
        matrix1 = np.loadtxt(f1)
    
    with open(file2_path, "r", encoding="utf-8") as f2:
        matrix2 = np.loadtxt(f2)
    
    if matrix1.shape != (size, size):
        raise ValueError(f"{file1_path}: expected {(size, size)} but got {matrix1.shape}")
    
    if matrix2.shape != (size, size):
        raise ValueError(f"{file2_path}: expected {(size, size)} but got {matrix2.shape}")
    
    return size, matrix1, matrix2

def read_result_matrix(file_path: str, size: int):
    with open(file_path, "r", encoding="utf-8") as f:
        data = np.loadtxt(f)

    if data.shape != (size, size):
        raise ValueError(f"{file_path}: expected {(size, size)} but got {data.shape}")

    return size, data

def multiply_matrices(matrix1, matrix2):
    return np.dot(matrix1, matrix2)

def check_one(size: int, base_dir=".", base_dir_result_mul="."):
    input_file_A = f"{base_dir}/matrixA_size_{size}.txt"
    input_file_B = f"{base_dir}/matrixB_size_{size}.txt"
    result_file = f"{base_dir_result_mul}/result_size_{size}.txt"

    in_size, A, B = read_file_to_matrix(input_file_A, input_file_B, size)
    out_size, C_cpp = read_result_matrix(result_file, size)

    if in_size != size or out_size != size:
        raise ValueError(f"Size mismatch. expected {size}, got input={in_size}, output={out_size}")

    C_np = multiply_matrices(A, B)

    return np.array_equal(C_cpp.astype(int), C_np.astype(int))

def main():
    sizes = [200, 400, 800, 1200, 1600, 2000]

    all_ok = True
    for s in sizes:
        try:
            ok = check_one(s, base_dir="Data_Matrix", base_dir_result_mul="Data_mul_matrix")
            print(f"Size {s}: {'PASS' if ok else 'FAIL'}")
            all_ok = all_ok and ok
        except Exception as e:
            print(f"Size {s}: ERROR -> {e}")
            all_ok = False

    if all_ok:
        print("\n🔥 Congratulation, all tests were passed!")
    else:
        print("\n❌ Some tests failed!")

if __name__ == "__main__":
    main()