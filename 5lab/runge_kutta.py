import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def f(x, y):
    return 2*y + x

def eq(x):
    return (np.exp(2*x) - 2 * x - 1) / 4

def runge_kutta(f, x0, y0, x_end, N, T):
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
N = 41 # Corresponding to T = 0.4
#N = 41  #T = 0.2
T = (x_end - x0) / (N - 1)

x_rk, y_rk = runge_kutta(f, x0, y0, x_end, N, T)


y_actual = eq(x_rk)
error = np.abs(y_actual - y_rk)


data = {
    "T": np.full_like(x_rk, (x_end - x0) / (N - 1)),
    "x": x_rk,
    "y_actual": y_actual,
    "y_rk": y_rk,
    "error": error
}
df = pd.DataFrame(data)
print(df)

plot_results(x_rk, y_actual, y_rk, "Runge-Kutta metodas")

# Runge Kutta - antros eiles metodas, nes jei sumazinsim T 2 kartus, tai paklaida sumazes 2^2^2 = 4 8 kartus
# Kai T = 0.2, tai paklaida kai x = 0.4 yra 0.008785
# Kai T = 0.4 tai paklaida kai x = 0.4 yra 0.02638