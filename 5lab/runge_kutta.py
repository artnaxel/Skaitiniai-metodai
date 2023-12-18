import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def f(x, y):
    return 2*y + x

def eq(x):
    return (np.exp(2*x) - 2 * x - 1) / 4

def heuns_runge_kutta(f, x0, y0, x_end, N, T):
    x = np.linspace(x0, x_end, N)
    y = np.zeros(N)
    y[0] = y0

    for i in range(N - 1):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + T, y[i] + T * k1)
        y[i+1] = y[i] + (T / 2) * (k1 + k2)
    return x, y

def plot_results(x, y_actual, y_approx, method_name):
    plt.figure(figsize=(12, 6))
    plt.plot(x, y_actual, label='Tikslus sprendinys', linestyle='--')
    plt.plot(x, y_approx, label=f"{method_name}", marker='o', markersize=3)
    plt.title(f"{method_name} vs Tikslus sprendinys")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim(0, 50)
    plt.legend()
    plt.grid(True)
    plt.show()

x0 = 0.0
y0 = 0.0
x_end = 8.0
N = 40 # T = 0.4
#N = 20  #T = 0.2
T = (x_end - x0) / N

x_rk, y_rk = heuns_runge_kutta(f, x0, y0, x_end, N + 1, T)


y_actual = eq(x_rk)
error = np.abs(y_actual - y_rk)


data = {
    "T": T,
    "x": x_rk,
    "y_actual": y_actual,
    "y_rk": y_rk,
    "error": error
}
df = pd.DataFrame(data)
print(df)

plot_results(x_rk, y_actual, y_rk, "Runge-Kutta metodas")