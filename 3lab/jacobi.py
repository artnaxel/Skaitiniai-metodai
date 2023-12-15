import numpy as np
import time

def jacobi_convergence_condition(A):
    n = A.shape[0]
    
    for i in range(n):
        diagonal_element = abs(A[i, i])
        sum_ratio = sum(abs(A[i, j]) / diagonal_element for j in range(n) if j != i)

        if sum_ratio >= 1:
            return False
    return True

def create_matrix_A(N):
    A = np.zeros((N, N))
    
    np.fill_diagonal(A, 3)
    
    np.fill_diagonal(A[0:-1, 1:], -1)
    
    np.fill_diagonal(A[1:, 0:-1], -1)
    
    return A

def jacobi_method(A, N, eps=1e-6, max_iter=1000):
    x = np.zeros(N)
    D = np.diag(np.diag(A))
    R = A - D
    D_inv = np.linalg.inv(D)
    
    for _ in range(max_iter):
        F = 1 / ((N+1)**2) * (x**2 + 1)
        x_new = np.dot(D_inv, F - np.dot(R, x))
        error = np.sum(np.abs(x_new - x))
        if error < eps:
            break 
        x = x_new

    return x_new

def solve_system(N, eps=1e-8, max_iter=1000):
    A = create_matrix_A(N)
    
    if not jacobi_convergence_condition(A):
        print("Jakobio metodas nekonverguoja!")
        return None
    start_time = time.time()
    X = jacobi_method(A, N, eps, max_iter)
    end_time = time.time()
    print(f"{'Sprendimo laikas:':<12}{end_time - start_time:<20.8f}")

    return X

N = 20
solution = solve_system(N)
print("Sprendimas:", solution)
