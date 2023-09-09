import numpy as np
import math

def generate_and_filter_odd_numbers(N):
    positive_numbers = np.arange(1, N + 1)
    odd_numbers = positive_numbers[positive_numbers % 2 != 0]
    return odd_numbers

def generate_and_replace_neg_numbers(N):
    numbers = np.arange(-1, N - 1)
    numbers = np.where(numbers < 0, 0, numbers)
    return numbers

def generate_3d_points(N):
    return np.random.rand(3, N);

def calculate_distances(X, Y):
    diff = X - Y
    distances = np.linalg.norm(diff, axis=0)
    return distances

N = 3

result_odd = generate_and_filter_odd_numbers(N)
result_neg = generate_and_replace_neg_numbers(N)

print("Returning odd numbers:\n")
print(result_odd)

print("Returning negative replaced with 0:\n")
print(result_neg)

X = generate_3d_points(N)
Y = generate_3d_points(N)

print("\nFirst set of points:")
print(X)

print("\nSecond set of points:")
print(Y)

distances = calculate_distances(X, Y)
print("\nDistances: ")
print(distances)
