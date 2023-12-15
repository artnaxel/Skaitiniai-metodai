import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def f(x):
    return x**2 * np.cos(x**3)

def f_derivative(x):
    return 2*x*np.cos(x**3) - 3*x**5*np.sin(x**3)

def integrand(x):
    return x**2 * np.cos(x**3)

a = 0
b = np.pi / 4

def adaptive_left_rectangle(func, a, b, tol):
    count = 0
    def recursive_integration(a, b, integral_old, tol):
        nonlocal count
        mid = (a + b) / 2
        left_integral = left_rectangle_rule(func, a, mid, 1)
        right_integral = left_rectangle_rule(func, mid, b, 1)
        integral_new = left_integral + right_integral
        count += 1
        if abs(integral_new - integral_old) < tol:
            return integral_new
        else:
            return (recursive_integration(a, mid, left_integral, tol/2) + 
                    recursive_integration(mid, b, right_integral, tol/2))

    initial_integral = left_rectangle_rule(func, a, b, 1)
    final_integral = recursive_integration(a, b, initial_integral, tol)
    return final_integral, count

def left_rectangle_rule(func, a, b, n):
    h = (b - a) / n
    return sum(func(a + i * h) for i in range(n)) * h

tolerance = 1e-6

final_integral, count = adaptive_left_rectangle(f, a, b, tolerance)
print(count)
integral, error = quad(integrand, a, b)

print("Adaptyvaus algoritmo rezultatas:", final_integral)
print("Scipy rezultatas:", integral)
