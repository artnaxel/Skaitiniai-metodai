import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def f(x):
    return x**1.5 * np.cos(x)

def absolute_error(actual, approx):
    return np.abs(actual - approx)

def compute_quadratic_coefficients(nodes, values):
    coefficients = []
    for i in range(0, len(nodes) - 2, 2):
        x0, x1, x2 = nodes[i], nodes[i+1], nodes[i+2]
        y0, y1, y2 = values[i], values[i+1], values[i+2]
        f_01 = (y1 - y0) / (x1 - x0)
        f_012 = ((y2 - y1) / (x2 - x1) - f_01) / (x2 - x0)

        a = f_012
        b = f_01 - f_012 * (x0 + x1)
        c = y0 - f_01 * x0 + f_012 * x0 * x1
        coefficients.append((a, b, c))
    print(len(coefficients))
    return coefficients

def compute_interpolation_error(x, nodes, coefficients, values):
    for i in range(0, len(nodes) - 2, 2):
        if nodes[i] <= x <= nodes[i+2]:
            _, _, f_012 = coefficients[i // 2]
            x0, x1, x2 = nodes[i], nodes[i+1], nodes[i+2]
            print(x0, x1, x2)
            y1, y2 = values[i+1], values[i+2]
            break

    if i + 3 < len(nodes):
        x3 = nodes[i + 3]
        y3 = f(x3)
    else:
        x3 = nodes[i - 1]
        y3 = f(x3)

    f_0123 = (((y3 - y2) / (x3 - x2) - (y2 - y1) / (x2 - x1)) / (x3 - x1) - f_012) / (x3 - x0)

    M3 = np.abs(6 * f_0123)

    return (1/6) * M3 * np.abs((x - x0) * (x - x1) * (x - x2))

def quadratic_interpolation(x, nodes, coefficients):
    for i in range(0, len(nodes) - 2, 2):
        if nodes[i] <= x <= nodes[i+2]:
            a, b, c = coefficients[i // 2]
            return a * x**2 + b * x + c

    return None


a, b = 2, 10
n_intervals = 6
x = 9

nodes = np.linspace(a, b, n_intervals + 1)
values = f(nodes)

table_of_values = np.column_stack((nodes, values))

coefficients = compute_quadratic_coefficients(nodes, values)
interp_value = quadratic_interpolation(x, nodes, coefficients)

error_value = compute_interpolation_error(x, nodes, coefficients, values)

print('Reikšmiu lentelė:\n', table_of_values)
print(f'Interpoliavimo reikšmė taške {x}: ', interp_value)
print('Tikroji reiksme: ', f(x))
print('Paklaidos ivertis: ', error_value)
print('Paklaida: ', absolute_error(f(x), interp_value))

x_plot = np.linspace(a, b, 10000)
y_actual = f(x_plot)
print(coefficients)
y_interp = np.array([quadratic_interpolation(xi, nodes, coefficients) for xi in x_plot])

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_actual, label="Funkcija")
plt.plot(x_plot, y_interp, label="Interpoliacinis Polinomas", linestyle="--")
plt.scatter(nodes, values, color='orange', label="Interpoliacijos Mazgai")
plt.scatter(x, interp_value, color='red', label=f"Interpoliuota reikšmė taške {x}", zorder=5)
for i, txt in enumerate(nodes):
    plt.annotate(f'({txt:.1f}, {values[i]:.2f})', (nodes[i], values[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='orange')

plt.annotate(f'({x}, {interp_value:.2f})', (x, interp_value), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='black')

plt.xlabel("x")
plt.ylabel("y")
plt.title("Kvadratinis interpoliacinis polinomas")
plt.legend()
plt.grid(True)
plt.show()

