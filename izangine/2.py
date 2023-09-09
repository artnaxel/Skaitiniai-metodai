import numpy as np

def generate_matrix(N):
    matrix = np.zeros((2 * N + 1, 2 * N + 1), dtype=int)

    matrix[:N, :N] = 3

    matrix[:N, N + 1:] = np.arange(1, N + 1)

    matrix[N + 2::2, :N:2] = 1
    matrix[N + 1::2, 1:N:2] = 1

    matrix[N + 1::3, N + 1:] = 2
    matrix[N + 3::3, N + 1:] = 2

    matrix[N + 1::3, N + 2::3] = 0
    matrix[N + 3::3, N + 2::3] = 0

    return matrix

N = 8
result = generate_matrix(N)
print(result)