import numpy as np

def generate_tridiagonal_matrix(N):
    main_diag = np.random.choice([ 1, 2], N)
    diagonal_matrix = np.diag(main_diag)
    upper_diag = np.random.choice([1, 2], N - 1)
    lower_diag = np.random.choice([ 1, 2], N - 1)
    
    tridiagonal_matrix = diagonal_matrix + np.diag(upper_diag, k=1) + np.diag(lower_diag, k=-1)
    
    return tridiagonal_matrix

N = 5
tridiagonal_matrix = generate_tridiagonal_matrix(N)
print(tridiagonal_matrix)