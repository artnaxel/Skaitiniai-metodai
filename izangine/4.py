import numpy as np
import matplotlib.pyplot as plt

A = 10
B = 11
a = 0.0
b = 2.0
c = 1.5

def f(x):
    return np.sin(A * np.pi * x) + B * x

x = np.linspace(a, b, 1000)
y = f(x)

# Isvestine
slope_at_C = A * np.pi * np.cos(A * np.pi * c) + B

# Liestine
tangent_line = slope_at_C * (x - c) + f(c)

graph_color = 'royalblue'
tangent_color = 'red'

plt.figure(figsize=(12, 8))
plt.plot(x, y, label=f'$f(x) = \sin({A}\pi x) + {B}x$', color=graph_color)
plt.plot(x, tangent_line, label=f'Tangent at C ($A={A}$, $B={B}$, $c={c}$)', color=tangent_color)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.xlim((1.35, 1.6), )
plt.title('Plot of $f(x) = \sin(A\pi x) + Bx$ with Tangent at C')
plt.legend()
plt.grid(True)
plt.show()
