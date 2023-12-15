import numpy as np
from scipy.linalg import lu_factor, lu_solve
import time

N = 8000
def create_matrix_A(N):
    A = np.zeros((N, N))
    
    np.fill_diagonal(A, 3)
    
    np.fill_diagonal(A[0:-1, 1:], -1)
    
    np.fill_diagonal(A[1:, 0:-1], -1)

    return A

def F(X, c):
    N = len(X)
    F = np.zeros(N)
    for i in range(N):
        F[i] = c * (X[i]**2 + 1)
    return F


X = np.zeros(N)

epsilon = 1e-6

start_time = time.time()

A = create_matrix_A(N)

while True:
    c = 1 / (N + 1)**2
    FX = F(X, c)
    lu, piv = lu_factor(A)
    X_new = lu_solve((lu, piv), FX)
    error = np.max(np.abs(X_new - X))
    
    X = X_new

    if error < epsilon:
        break

end_time = time.time()

print("Solution X:", X)

print("Execution time: {:.2f} seconds".format(end_time - start_time))  # Print the total execution time
