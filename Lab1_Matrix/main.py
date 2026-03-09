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


def check_one(size: int, base_dir="."):
    input_file = f"{base_dir}/matrix_size_{size}.txt"
    result_file = f"{base_dir}/result_size_{size}.txt"

    in_size, A, B = read_file_to_matrix(input_file)
    out_size, C_cpp = read_result_matrix(result_file)

    if in_size != size or out_size != size:
        raise ValueError(f"Size mismatch. expected {size}, got input={in_size}, output={out_size}")

    C_np = multiply_matrices(A, B)

    return np.array_equal(C_cpp.astype(int), C_np.astype(int))


def plot_timing(timing_file="timing_results.txt"):
    df = pd.read_csv(timing_file)

    df["Size"] = pd.to_numeric(df["Size"], errors="coerce")
    df["Time_ms"] = pd.to_numeric(df["Time_ms"], errors="coerce")

    df = df.dropna(subset=["Size", "Time_ms"]).sort_values("Size")

    plt.figure()
    plt.plot(df["Size"], df["Time_ms"], marker="o")
    plt.xticks(df["Size"])
    plt.xlabel("Matrix Size")
    plt.ylabel("Time (ms)")
    plt.title("Matrix Multiplication Time vs Size")
    plt.grid(True)
    plt.savefig("graphic_time.png")
    plt.show()


def main():
    sizes = [200, 400, 800, 1200, 1600, 2000]

    all_ok = True
    for s in sizes:
        try:
            ok = check_one(s, base_dir=".")
            print(f"Size {s}: {'PASS' if ok else 'FAIL'}")
            all_ok = all_ok and ok
        except Exception as e:
            print(f"Size {s}: ERROR -> {e}")
            all_ok = False

    if all_ok:
        print("Congratulation, all tests were passed!")
    else:
        print("Some tests failed!")

    plot_timing("timing_results.txt")


if __name__ == "__main__":
    main()