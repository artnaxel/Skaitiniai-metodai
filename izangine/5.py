import numpy as np

def generate_and_filter_odd_numbers(N):
    positive_numbers = np.arange(1, N + 1)
    odd_numbers = positive_numbers[positive_numbers % 2 != 0]
    return odd_numbers

def generate_and_replace_neg_numbers(N):
    numbers = np.arange(-1, N - 1)
    numbers = np.where(numbers < 0, 0, numbers)
    return numbers

def generate_3d_points(N):
    return np.random.rand(3, N)

def calculate_distances(X, Y):
    diff = X - Y
    distances = np.linalg.norm(diff, axis=0)
    return distances

def generate_random_vector(N):
    return np.random.randint(-100, 100, N)

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

count_greater_than_previous = np.sum(random_vector[1:] > random_vector[:-1])

count_sign_changes = np.sum(np.sign(random_vector[1:]) != np.sign(random_vector[:-1]))

print("\nRandom vector:", random_vector)
print("Greater than previous:", count_greater_than_previous)
print("Sign changes:", count_sign_changes)
