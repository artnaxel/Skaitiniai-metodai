import numpy as np

# Sugeneruokite veiktorių iš N sveikų teigiamų skaičių. Grąžinkite nelyginius skaičius.
def generate_and_filter_odd_numbers(N):
    positive_numbers = np.arange(1, N + 1)
    odd_numbers = positive_numbers[positive_numbers % 2 != 0]
    return odd_numbers

# Sugeneruokite vektorių iš N skaičių. Pakeiskite neigiamus skaičius į 0.
def generate_and_replace_neg_numbers(N):
    numbers = np.arange(-1, N - 1)
    numbers = np.where(numbers < 0, 0, numbers)
    return numbers

# Sugeneruokite 2 trimačių taškų rinkinius. Juos saugokite matricose X ir Y.
def generate_3d_points(N):
    x_coords = np.random.randint(1, 10 + 1, size=N)
    y_coords = np.random.randint(1, 10 + 1, size=N)
    z_coords = np.random.randint(1, 10 + 1, size=N)
    
    points = np.vstack((x_coords, y_coords, z_coords))
    
    return points

# Grąžinkite atstumus tarp taškų.
def calculate_distances(X, Y):
    distances = np.sqrt(np.sum((Y - X)**2, axis=1))
    return distances

# Sugeneruokite vektorių iš N skaičių
def generate_random_vector(N):
    return np.random.randint(-100, 100, N)

# Suskaičiuokite kiek kartų elementas buvo didesnis už prieš tai buvusį
def count_greater_than_previous():
    return np.sum(random_vector[1:] > random_vector[:-1])

# Suskaičiuokite kiek kartų seka keitė ženklą
def count_sign_changes():
    return np.sum(np.sign(random_vector[1:]) != np.sign(random_vector[:-1]))

N = 3

result_odd = generate_and_filter_odd_numbers(N)
result_neg = generate_and_replace_neg_numbers(N)

print("\nReturning odd numbers:", result_odd)

print("\nReturning negative replaced with 0:", result_neg)

X = generate_3d_points(N)
Y = generate_3d_points(N)

print("\nFirst set of points:\n", X)

print("\nSecond set of points:\n", Y)

distances = calculate_distances(X, Y)
print("\nDistances: ", distances)

random_vector =  generate_random_vector(N)

print("\nRandom vector:", random_vector)
print("Greater than previous:", count_greater_than_previous())
print("Sign changes:", count_sign_changes())
