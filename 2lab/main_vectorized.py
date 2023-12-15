import numpy as np
import time

def create_matrix_A(N):
    A = np.zeros((N, N))
    
    np.fill_diagonal(A, 2)
    
    np.fill_diagonal(A[0:-1, 1:], -1)
    
    np.fill_diagonal(A[1:, 0:-1], -1)

    return A

def lu_decomposition(A):
    n = A.shape[0]
    L = np.eye((n))
    U = np.copy(A)
    
    for i in range(n):
        
        for k in range(i+1, n):

            c = -U[k, i] / U[i, i]

            U[k, i:] += c * U[i, i:]

            L[k, i] = -c
    
    return L, U

def forward_substitution(L, B):
    n = L.shape[0]
    y = np.zeros(n)

    y[0] = B[0] / L[0, 0]

    for i in range(1, n):
        y[i] = (B[i] - (L[i, :i] @ y[:i])) / L[i, i]
    return y

def backward_substitution(U, y):
    n = U.shape[0]
    x = np.zeros(n)

    x[-1] = y[-1] / U[-1, -1]

    for i in range(n-2, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i:], x[i:])) / U[i, i]
    return x

def solve_system(N, eps=1e-6, max_iter=1000):
    A = create_matrix_A(N)
    X = np.zeros(N)

    timeLU = 0
    timeSolution = 0

    start_time_LU = time.time()

    L, U = lu_decomposition(A)

    end_time_LU = time.time()

    timeLU = end_time_LU - start_time_LU

    for _ in range(max_iter):

        F = 1 / ((N+1)**2) * (X**2 + 1)

        start_time_solution = time.time()

        y = forward_substitution(L, F)
        X_new = backward_substitution(U, y)

        end_time_solution = time.time()

        timeSolution += end_time_solution - start_time_solution
        error = np.max(np.abs(X_new - X))
        if error < eps:
            break
        X = X_new

    print(f"{'Iteracija':<12}{'LU Dekompozicija':<20}")
    print('-' * 52)
    print(f"{'Dekompozicijos laikas:':<12}{timeLU:<20.8f}")
    print(f"{'Sprendimo laikas:':<12}{timeSolution:<20.8f}")

    return X_new

N = 20
solution = solve_system(N)
print("Sprendimas:", solution)
