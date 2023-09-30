import matplotlib.pyplot as plt
import numpy as np

def g(x):
    return np.exp(-x**2)

def dg(x):
    return -2 * x * np.exp(-x**2)

def find_q():
    x = np.linspace(a, b, 1000)
    return np.max(dg(x))

def simple_iteration(x0, tol=1e-6, max_iter=1000):
    x_prev = x0
    iterations = []
    q = find_q()
    for i in range(max_iter):
        x_next = g(x_prev)
        error = abs(x_next - x_prev)
        iterations.append((i, x_next, error))

        if error <= ((1-q) / q) * tol:
            return x_next, iterations
        x_prev = x_next

    raise ValueError("Nekonverguoja!")


x0 = 0.5
a = -1
b = 1
root, iterations = simple_iteration(x0)

# Iteration table
print(f"{'Iteracija':<10}{'Artinys':<20}{'Paklaida':<20}")
for i, x, error in iterations:
    print(f"{i:<10}{x:<20.6f}{error:<20.6f}")

# Create a range of x values
x = np.linspace(a, b, 1000)

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
