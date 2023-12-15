import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import pandas as pd

def f(x):
    return x**2 * np.cos(x**3)

def f_derivative(x):
    return 2*x*np.cos(x**3) - 3*x**5*np.sin(x**3)

def integrand(x):
    return x**2 * np.cos(x**3)

a = 0
b = np.pi / 4

a, b = 0, np.pi/4
integral, error = quad(integrand, a, b)

def left_rectangle_rule(func, a, b, n):
    h = (b - a) / n
    return sum(func(a + i * h) * h for i in range(n))

nodes = [2, 20,  200]
approx_integral_values = []
approx_errors = []
exact_errors = []
h_values = []

x_values = np.linspace(a, b, 1000)
M1 = max(abs(f_derivative(x)) for x in x_values)

#  paklaida mazeja tiek kartu kiek mazeja h
for n in nodes:
    h = (b - a) / n
    h_values.append(h)
    approx_integral = left_rectangle_rule(f, a, b, n)
    approx_integral_values.append(approx_integral)

    approx_error = M1 * ((b - a) / 2) * h
    approx_errors.append(approx_error)

    exact_error = abs(integral - approx_integral)
    exact_errors.append(exact_error)


result_data = {
    "n": nodes,
    "h_reikšmės": h_values,
    "apytiksles_integralo_reiksmes": approx_integral_values,
    "apytiksles_paklaidos": approx_errors,
    "tiksliosios_paklaidos": exact_errors
}

df_results = pd.DataFrame(result_data)

df_formatted = df_results.copy()
df_formatted.columns = ['n', 'h reikšmė', 'Apytikslė integralo reikšmė', 'Apytikslė paklaida', 'Tikslioji paklaida']
df_formatted['n'] = df_formatted['n']
df_formatted['h reikšmė'] = df_formatted['h reikšmė'].map('{:,.5f}'.format)
df_formatted['Apytikslė integralo reikšmė'] = df_formatted['Apytikslė integralo reikšmė'].map('{:,.5f}'.format)
df_formatted['Apytikslė paklaida'] = df_formatted['Apytikslė paklaida'].map('{:,.5f}'.format)
df_formatted['Tikslioji paklaida'] = df_formatted['Tikslioji paklaida'].map('{:,.5f}'.format)

print(df_formatted)

plt.figure(figsize=(10, 6))
plt.plot(h_values, approx_errors, label='Paklaidos įvertis', marker='o')
plt.xlabel('h')
plt.ylabel('Paklaida')
plt.title('Paklaidų grafikai')
plt.legend()
plt.grid(True)
plt.show()
