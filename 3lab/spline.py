import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

def f(x):
    return np.power(x, 1.5) * np.cos(x)

def evaluate_spline(coefficients, intervals, x_values):
    if np.isscalar(x_values):
        x_values = np.array([x_values])

    y_values = np.zeros_like(x_values)
    for i in range(len(coefficients)):
        a, b, c = coefficients[i]
        start, end = intervals[i], intervals[i+1]
        mask = (x_values >= start) & (x_values <= end)
        y_values[mask] = a*x_values[mask]**2 + b*x_values[mask] + c
    
    return y_values if len(y_values) > 1 else y_values[0]

def setup_spline_system(x, y):
    n = len(x) - 1
    A = np.zeros((3*n, 3*n))
    B = np.zeros(3*n)

    for i in range(n):
        A[2*i][3*i] = x[i]**2
        A[2*i][3*i + 1] = x[i]
        A[2*i][3*i + 2] = 1
        B[2*i] = y[i]

        A[2*i + 1][3*i] = x[i+1]**2
        A[2*i + 1][3*i + 1] = x[i+1]
        A[2*i + 1][3*i + 2] = 1
        B[2*i + 1] = y[i+1]
        if i < n - 1:
            A[2*n + i][3*i] = 2*x[i+1]
            A[2*n + i][3*i + 1] = 1
            A[2*n + i][3*(i+1)] = -2*x[i+1]
            A[2*n + i][3*(i+1) + 1] = -1

    derivative_at_x0 = (y[1] - y[0]) / (x[1] - x[0])

    A[-1][0] = 2*x[0]
    A[-1][1] = 1
    B[-1] = derivative_at_x0
    print(A)
    return A, B

a, b = 2, 10
n_intervals = 4

nodes = np.linspace(a, b, n_intervals + 1)
values = f(nodes)

table_of_values = np.column_stack((nodes, values))
print('Table of values:\n', table_of_values)


x_plot = np.linspace(a, b, 1000)

A, B = setup_spline_system(nodes, values)

coefficients = np.linalg.solve(A, B)

spline_coefficients = coefficients.reshape(-1, 3)

x = 2.0
print(evaluate_spline(spline_coefficients, nodes, x))
spline_values = np.array(evaluate_spline(spline_coefficients, nodes, x_plot))

plt.figure(figsize=(10, 6))
# plt.xlim([5, 7])
# plt.ylim([0, 18])
plt.plot(x_plot, spline_values, label='Kvadratinis splainas', color='red')
plt.plot(x_plot, f(x_plot), label='Pradinė funkcija', linestyle='--', color='blue')
plt.scatter(nodes, values, color='green', label='Mazgai')
for i, txt in enumerate(nodes):
    plt.annotate(f'({txt:.1f}, {values[i]:.2f})', (nodes[i], values[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='orange')
plt.title('Kvadratinis splainas ir pradinė funkcija')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()


