import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def f(x, y):
    return 2*y + x

def eq(x):
    return (np.exp(2*x) - 2 * x - 1) / 4

def euler_method(f, x0, y0, T, x_end, N):
    x_values = np.linspace(x0, x_end, N)
    y_values = [y0]
    
    for i in range(1, N):
        y_new = y_values[-1] + T * f(x_values[i-1], y_values[-1])
        y_values.append(y_new)
    
    return x_values, y_values

def calculate_error(y_actual, y_approx):
    return np.abs(y_actual - np.array(y_approx))

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
N = 41
#N = 81 # T = 0.1
#N = 41  #T = 0.2
#N = 21  T = 0.4
#N = 11  T = 0.8
#N = 6   T = 1.6

T = (x_end - x0) / (N - 1)

x_euler, y_euler = euler_method(f, x0, y0, T, x_end, N)

y_actual = eq(x_euler)
error = calculate_error(y_actual, y_euler)


data = {
    "T": T,
    "x": x_euler,
    "y_actual": y_actual,
    "y_euler": y_euler,
    "error": error
}

df = pd.DataFrame(data)

print(df)
plot_results(x_euler, y_actual, y_euler, "Eulerio metodas")

# Eulerio metodas yra pirmos eiles metoads nes sumazinsim T 2 kartus, tai paklaida sumazes 4 kartus
# local error (error per step) is proportional to the square of the step size, and the global error (error at a given time) is proportional to the step size itself.
