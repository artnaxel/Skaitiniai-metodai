import numpy as np

# Sudarykite (2N + 1) x (2N + 1) matricą pagal pavyzdį
def generate_matrix(N):
    matrix = np.zeros((2 * N + 1, 2 * N + 1), dtype=int)
    
    # Užpildome iki N-osios eilutės, N-ojo stulpelio 3ejetais
    matrix[:N, :N] = 3
    
    #Iki N-osios eilutės ir nuo N + 1 stulpelio užpildome skaičias nuo 1 iki N + 1
    matrix[:N, N + 1:] = np.arange(1, N + 1)
    
    #Kas antra eilutę nuo N + 2 ir iki N-ojo stulpelio, kas antra stulpelį užpildome 1
    matrix[N + 2::2, :N:2] = 1
    #Kas antrą eilutę nuo N + 1, ir nuo 1ojo stulpelio kas antra stulpelį užpildome 1
    matrix[N + 1::2, 1:N:2] = 1

    #Kas trečią eilutę nuo N + 1 ir nuo N + 1 stulpelio užpildome 2
    matrix[N + 1::3, N + 1:] = 2
    #Kas trečią eilutę nuo N + 3 ir nuo N + 1 stulpelio užpildome 2
    matrix[N + 3::3, N + 1:] = 2

    #Kas trečią eilutę nuo N + 1 ir kas trečią stulpelį nuo N + 2 užpildome 0
    matrix[N + 1::3, N + 2::3] = 0
    #Kas trečią eilutę nuo N + 3 ir kas trečią stulpelį nuo N + 2 užpildome 0
    matrix[N + 3::3, N + 2::3] = 0

    return matrix

N = 9
result = generate_matrix(N)
print(result)