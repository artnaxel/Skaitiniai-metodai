import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return np.exp(-x**2) - x

def bisection(a, b, tol=1e-6, max_iter=1000):
    if f(a) * f(b) > 0:
        print("Netinkamas pradinis intervalas.")
        return None, [], []
    
    iteration = 0
    c = (a + b) / 2.0
    error = abs(b - a) / 2.0
    history = [c]
    errors = [error]
    while error > tol and iteration < max_iter:
        if f(c) == 0:
            break
        elif f(c) * f(a) < 0:
            b = c
        else:
            a = c
        c = (a + b) / 2.0
        error = abs(b - a) / 2.0
        iteration += 1
        history.append(c)
        errors.append(error)

    return c, history, errors

# Initial interval [a,b]
a = 0.64
b = 0.66
epsilon = 1e-8

root, iterations, errors = bisection(a, b, tol=epsilon)

# Iteration table
print("Iteracijos numeris | n-asis artinys | Paklaida")
print("---------------------------------------------------")
for i in range(0, len(iterations)):
    iter_num = i
    curr_approx = iterations[i]
    error = errors[i]
    print(f"{iter_num:^18} | {curr_approx:^12.6f} | {error:^24.8f}")

# Getting the y values corresponding to each x value for plotting
y_values = [f(i) for i in iterations]

# Plot function graph
x = np.linspace(min(iterations)-1, max(iterations)+1, 400)
y = f(x)
plt.figure(figsize=(10, 6))
plt.plot(x, y, label="Funkcijos grafikas")

# Mark specific points of iterations
specific_iterations = [0, 2, 4, 7, len(iterations)-1] 
for i in specific_iterations:
    plt.plot(iterations[i], y_values[i], 'ro')
    plt.text(iterations[i], y_values[i]+0.05, f'{iterations[i]:.6f} ({i})', fontsize=9)

margin = 0.4
plt.xlim(min(iterations)-margin, max(iterations)+margin)

plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.title("Pusiaukirtos metodo iteracijos ir funkcijos grafikas")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()
