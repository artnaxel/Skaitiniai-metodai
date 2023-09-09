import numpy as np
import matplotlib.pyplot as plt

A = 10
B = 11
a = 0.0
b = 2.0

def f(x):
    return np.sin(A * np.pi * x) + B * x

x = np.linspace(a, b, 1000)

y = f(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y, label=f'$f(x) = \sin({A}\pi x) + {B}x$')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Plot of $f(x) = \sin(A\pi x) + Bx$')
plt.legend()
plt.grid(True)
plt.show()
