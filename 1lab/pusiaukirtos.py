import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return np.exp(-x**2) - x

def bisection(a, b, tol=1e-6, max_iter=1000):
    if f(a) * f(b) > 0:
        print("Netinkamas pradinis intervalas.")
        return None, []
    
    iteration = 0
    c = (a + b) / 2.0
    history = [c]
    while abs(b - a) / 2.0 > tol and iteration < max_iter:
        if f(c) == 0:
            break
        elif f(c) * f(a) < 0:
            b = c
        else:
            a = c
        c = (a + b) / 2.0
        iteration += 1
        history.append(c)  # Changed this line as well

    return c, history

# Initial interval [a,b]
a = 0
b = 2
epsilon = 1e-6

root, iterations = bisection(a, b, tol=epsilon)
print("Iteracijos numeris | n-asis artinys | Paklaida")
print("---------------------------------------------------")
for i in range(1, len(iterations)):
    iter_num = i
    curr_approx = iterations[i]
    prev_approx = iterations[i - 1]
    error = abs(curr_approx - prev_approx)
    print(f"{iter_num:^18} | {curr_approx:^12.6f} | {error:^24.6f}")

# Getting the y values corresponding to each x value for plotting
y_values = [f(i) for i in iterations]

# Plot function graph
x = np.linspace(min(iterations)-1, max(iterations)+1, 400)
y = f(x)
plt.figure(figsize=(10, 6))
plt.plot(x, y, label="Function graph")

# Mark points found in each iteration
specific_iterations = [1, 2, 4, 6, len(iterations)-1] 
for i in specific_iterations:
    plt.plot(iterations[i], y_values[i], 'ro')
    plt.text(iterations[i], y_values[i]+0.05, f'{iterations[i]:.6f} ({i})', fontsize=9)

# Adjust the xlim according to the c value
margin = 0.01
plt.xlim(min(iterations)-margin, max(iterations)+margin)

plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.title("Bisection Method Iterations on Function Graph")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()
