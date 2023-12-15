import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def f(x, y):
    return 2*y + x

def eq(x):
    return (np.exp(2*x) - 2 * x - 1) / 4

x0 = 0.0
y0 = 0.0
x_end = 8.0
N = 81

T = (x_end - x0) / (N - 1)

x_euler = np.linspace(x0, x_end, N)
y_euler = [y0]

for i in range(1, N):
    y_new = y_euler[-1] + T * f(x_euler[i-1], y_euler[-1])
    y_euler.append(y_new)

y_actual = eq(x_euler)

error = np.abs(y_actual - np.array(y_euler))
data = {
    "T": T,
    "x": x_euler,
    "y_actual": y_actual,
    "y_euler": y_euler,
    "error": error
}

df = pd.DataFrame(data)
print(df)


plt.figure(figsize=(12, 6))
plt.plot(x_euler, y_actual, label='Tikslus sprendinys', linestyle='--')
plt.plot(x_euler, y_euler, label="Eulerio metodas", marker='o', markersize=3)
plt.title("Eulerio metodas ir tikslus sprendinys")
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(0, 50)
plt.legend()
plt.grid(True)
plt.show()
