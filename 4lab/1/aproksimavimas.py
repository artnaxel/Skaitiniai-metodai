import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('1/duom5.txt')

def cos_lambda(x, lambda_):
    return np.cos(lambda_ * x)

def sin_lambda(x, lambda_):
    return np.sin(lambda_ * x)

def cos_2lambda(x, lambda_):
    return np.cos(2 * lambda_ * x)

def sin_2lambda(x, lambda_):
    return np.sin(2 * lambda_ * x)

def approx_function(x, a0, a1, a2, a3, a4, lambda_):
    return a0 + a1 * np.cos(lambda_ * x) + a2 * np.sin(lambda_ * x) + a3 * np.cos(2 * lambda_ * x) + a4 * np.sin(2 * lambda_ * x)

x, y = data[:, 0], data[:, 1]
P = 3.05
lambda_est = (2 * np.pi) / P
X = np.column_stack([
    np.ones_like(x), 
    cos_lambda(x, lambda_est),
    sin_lambda(x, lambda_est),
    cos_2lambda(x, lambda_est),
    sin_2lambda(x, lambda_est)
])
print(X)

X_T = X.T

A = np.dot(X_T, X)
b = np.dot(X_T, y)

a = np.linalg.solve(A, b)
print(a)

a0, a1, a2, a3, a4 = a

y_predicted = approx_function(x, a0, a1, a2, a3, a4, lambda_est)

x_values = np.linspace(x.min(), x.max(), 1000)

y_values = approx_function(x_values, a0, a1, a2, a3, a4, lambda_est)

error = np.sum((y - y_predicted) ** 2)
print('Paklaida sumavimo kvadratu normoje: ', error)

plt.scatter(x, y, label='Duoti taškai')
plt.plot(x_values, y_values, color='red', label='Aproksimuota funkcija')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Aproksimuota funkcija ir duoti taškai')
plt.legend()
plt.grid()
plt.show()


print("Koeficientai:", a)