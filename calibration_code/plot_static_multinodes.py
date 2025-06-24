import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Load data
df = pd.read_csv("calibration_data/static_test_4throw.csv")
df['FT_Z'] = -df['FT_Z'] / 4  # divide by 16

# Group by step
grouped = df.groupby('Step')

# --- Use S0 only ---
baseline = 726  # adjust if needed
avg_ft = grouped['FT_Z'].mean()
avg_sensor = grouped['S7'].mean() - baseline
std_sensor = grouped['S7'].std()


# Define linear threshold
limit = 15
mask = avg_ft < limit
sensor = avg_sensor[mask]
sensor_err = std_sensor[mask]
ft = avg_ft[mask]
x_model = np.linspace(ft.min(), ft.max(), 200)

sensor_fit = np.linspace(0, 300, 500)

# Linear fit
coeffs_linear = np.polyfit(sensor, ft, deg=1)
print("Linear Fit Coeffs:", coeffs_linear)
linear_fit = coeffs_linear[0] * sensor_fit + coeffs_linear[1]

# Quadratic fit
coeffs_quad = np.polyfit(sensor, ft, deg=2)
print("Quadratic Fit Coeffs:", coeffs_quad)
quad_fit = coeffs_quad[0] * sensor_fit**2 + coeffs_quad[1] * sensor_fit + coeffs_quad[2]

# Power fit
def power_func(x, a, b):
    return a * x**1.47 + b
params_power, _ = curve_fit(power_func, sensor, ft)
a, b = params_power
power_fit = power_func(sensor_fit, a, b)
print("Power Fit: a =", a, ", b =", b)

# Exponential fit
log_ft = np.log(ft)
b_exp, ln_a_exp = np.polyfit(sensor, log_ft, deg=1)
a_exp = np.exp(ln_a_exp)
exp_fit = a_exp * np.exp(b_exp * sensor_fit)

# Plotting
plt.figure(figsize=(8, 6))
plt.errorbar(
    x=avg_ft,
    y=avg_sensor,
    yerr=std_sensor,
    fmt='o',
    capsize=5,
    ecolor='gray',
    elinewidth=1,
    markersize=6,
    label='Step Avg (S0)'
)

plt.plot(linear_fit, sensor_fit, label="Linear Fit", color="red")
# Optional: add other fits
# plt.plot(quad_fit, sensor_fit, label="Quadratic Fit", color="orange")
# plt.plot(power_fit, sensor_fit, label="Power Fit", color="purple")
# plt.plot(exp_fit, sensor_fit, label="Exponential Fit", color="green")

plt.xlabel('FT Z Force / 2')
plt.ylabel('Average S0 Value (Baseline Removed)')
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.ylim(0, 100)
# plt.xlim(0, 10)
plt.title("S0 vs FT_Z/2")
plt.show()
