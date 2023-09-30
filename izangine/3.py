import numpy as np
import timeit

def generate_tridiagonal_matrix(N):
    main_diag = np.random.choice([0, 1, 2], N)
    diagonal_matrix = np.diag(main_diag)
    upper_diag = np.random.choice([0, 1, 2], N - 1)
    lower_diag = np.random.choice([0, 1, 2], N - 1)
    
    tridiagonal_matrix = diagonal_matrix + np.diag(upper_diag, k=1) + np.diag(lower_diag, k=-1)
    
    return tridiagonal_matrix

def generate_matrix(N, M):
    matrix = np.ones((N, M))
    return matrix

# Saduginkite naudodami 3 for ciklus
def multiply_matrices(matrix_A, matrix_B):
    if matrix_A.shape[1] == matrix_B.shape[0]:
    
        result_matrix = np.zeros((matrix_A.shape[0], matrix_B.shape[1]))
    
        for i in range(matrix_A.shape[0]):
            for j in range(matrix_B.shape[1]):
                for k in range(matrix_A.shape[1]):
                    result_matrix[i][j] += matrix_A[i][k] * matrix_B[k][j]
    
        return result_matrix
    else:
        print("Matrix multiplication is not possible")

# Sudauginkite naudojant numpy vektorių daugybą
def multiply_matrices_loop(matrix_A, matrix_B):
    if matrix_A.shape[1] == matrix_B.shape[0]:
    
        result_matrix = np.zeros((matrix_A.shape[0], matrix_B.shape[1]))
    
        for i in range(matrix_A.shape[0]):
            for j in range(matrix_B.shape[1]):
                result_matrix[i][j] = np.dot(matrix_A[i], matrix_B[:, j])
    
        return result_matrix
    else:
        print("Matrix multiplication is not possible")

# Panaudokite numpy operaciją
def multiply_matrices_np(matrix_A, matrix_B):
    if matrix_A.shape[1] == matrix_B.shape[0]:
        result_matrix = np.matmul(matrix_A, matrix_B)
        return result_matrix
    else:
        print("Matrix multiplication is not possible")
        
N = 20
M = 20

matrix_A = generate_tridiagonal_matrix(N)
matrix_B = generate_matrix(N, M)

print("Matrix A:\n", matrix_A)
print("Matrix B:\n", matrix_B)

result_matrix = multiply_matrices(matrix_A, matrix_B)
print("Result of multiply_matrices:")
print(result_matrix)

result_matrix_optimized = multiply_matrices_loop(matrix_A, matrix_B)
print("\nResult of multiply_matrices_loop:")
print(result_matrix_optimized)

result_matrix_np = multiply_matrices_np(matrix_A, matrix_B)
print("\nResult of multiply_matrices_np:")
print(result_matrix_np)

# Laiko matavimai
time_multiply_matrices = timeit.timeit(lambda: multiply_matrices(matrix_A, matrix_B), number=1000)

time_multiply_matrices_loop = timeit.timeit(lambda: multiply_matrices_loop(matrix_A, matrix_B), number=1000)

time_multiply_matrices_np = timeit.timeit(lambda: multiply_matrices_np(matrix_A, matrix_B), number=1000)

print(f"Time taken by multiply_matrices: {time_multiply_matrices * 1000:.2f} milliseconds")
print(f"Time taken by multiply_matrices_loop: {time_multiply_matrices_loop * 1000:.2f} milliseconds")
print(f"Time taken by multiply_matrices_np: {time_multiply_matrices_np * 1000:.2f} milliseconds")