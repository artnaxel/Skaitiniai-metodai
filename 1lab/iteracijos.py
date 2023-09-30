import matplotlib.pyplot as plt
import numpy as np

def val_error():
    return "Funkcija nekonverguoja!"
def g(x):
    return np.exp(-x**2)

def dg(x):
    return -2 * x * np.exp(-x**2)

def find_q(x):
    return np.max(np.abs(dg(x)))

def simple_iteration(x0, x, tol=1e-6, max_iter=1000):
    q = find_q(x)
    if ((dg(x) <= q) & (q < 1)).all():
        x_prev = x0
        iterations = []
        for i in range(max_iter):
            x_next = g(x_prev)
            error = abs(x_next - x_prev)
            iterations.append((i, x_next, error))

            if error <= ((1-q) / q) * tol:
                return x_next, iterations
            x_prev = x_next

        raise ValueError(val_error())
    else:
        raise ValueError(val_error())


x0 = 0.5
a = -1
b = 1
x = np.linspace(a, b, 1000)
root, iterations = simple_iteration(x0, x)

# Iteration table
print(f"{'Iteracija':<10}{'Artinys':<20}{'Paklaida':<20}")
for i, val, error in iterations:
    print(f"{i:<10}{val:<20.6f}{error:<20.6f}")

# Calculate y values based on the function y=e^(-x^2)
y1 = np.exp(-x**2)

# Calculate y values based on the derivative y=-2xe^(-x^2)
y2 = -2 * x * np.exp(-x**2)

# Plot y = e^(-x^2)
plt.plot(x, y1, label='y = e^(-x^2)')

# Plot y = x
plt.plot(x, x, label='y = x')

# Plot y = -2xe^(-x^2), the derivative
plt.plot(x, y2, label="y = -2xe^(-x^2)", linestyle='--')

# Mark specific points of iterations
selected_iterations = [0, 10, len(iterations) - 1]  
for i in selected_iterations:
    iter_num, x, _ = iterations[i]
    plt.plot(x, g(x), 'go')  
    plt.annotate(f'({x:.2f})', (x, g(x)), textcoords="offset points", xytext=(0,10), ha='center')

# Labels and titles
plt.title('Funkcijos grafikas ir tam tikrų iteracijų taškai')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
