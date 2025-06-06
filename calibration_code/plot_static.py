import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Load the collected step data
df = pd.read_csv("static_test_2nd_sensor.csv")
df['FT_Z'] = -df['FT_Z']

# Group by each step
grouped = df.groupby('Step')

# Define the baseline
baseline = 248.74000677506774

# Compute averages and std devs
avg_ft = grouped['FT_Z'].mean()
avg_sensor = grouped['Sensor'].mean()
avg_sensor = avg_sensor - baseline
std_sensor = grouped['Sensor'].std()

# Define the linear region threshold 
limit = 15
mask = avg_ft < limit
sensor = avg_sensor[mask]
sensor_err = std_sensor[mask]
ft = avg_ft[mask]
x_model = np.linspace(ft.min(), ft.max(), 200)

sensor_fit = np.linspace(0, 500, 500)
# Try to fit the linear function y = ax + b
coeffs_linear = np.polyfit(sensor, ft, deg = 1)
print(coeffs_linear) # [ 0.00883526 -1.07748348]
linear_fit = 0.00883526 * sensor_fit -1.07748348

# Try to fit the quadratic model y = a + b*x + c*x^2, x is sensor
coeffs_quad = np.polyfit(sensor, ft, deg =2)
print(coeffs_quad) # [ 3.42089499e-05 -8.71522057e-03  4.62796956e-01]
# Force = 0.00003497⋅x^2 − 0.002357⋅x + 0.10235
quad_fit = 3.42089499e-05 * (sensor_fit**2) -8.71522057e-03 * sensor_fit + 4.62796956e-01

# Try to fit the 1.47 power function
def power_func(x, a, b):
    return a * x**1.47 + b
params_power, _ = curve_fit(power_func, sensor, ft)
a, b = params_power
power_fit = a * (sensor_fit**1.47) + b
# Force = 0.0004731482094181381 * (sensor_fit**1.47) -0.6678598570322021
print(a)
print(b)

# Try to fit the exponential model y = a * e^(bx)
# log(Force) = ln(a) + b*Sensor
log_ft = np.log(ft)
coeffs = np.polyfit(sensor, log_ft, deg=1)
b = coeffs[0]
ln_a = coeffs[1]
a = np.exp(ln_a)
print(b)
print(ln_a)
# ln(Force) = -3.1994982717702247 + 0.009745754189733688 * Sensor

log_force = ln_a + b * sensor_fit
exp_fit = np.exp(log_force)

# # # Try to fit the ligatithmic model y = a + b*ln(x)
# x_fit = np.linspace(avg_ft.min(), avg_ft.max(), 1000)
# def log_func(x, a, b):
#     return a + b * np.log(x)
# params, _ = curve_fit(log_func, avg_ft, avg_sensor)
# a, b = params
# y_log = log_func(x_fit, a, b)
# print(a, b)

# Plot
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
    label='Step Avg'
)

# plt.plot(x_fit, y_log)
plt.plot(quad_fit, sensor_fit, label = "Quadratic Fit", color = "Orange")
plt.plot(exp_fit, sensor_fit, label = "Exp Fit", color = "Green")
plt.plot(linear_fit, sensor_fit, label = "Linear Fit", color = "Red")
plt.plot(power_fit, sensor_fit, label = "Power Fit", color = "Purple")
plt.ylim(0, 600)
plt.xlim(0, 16)
# plt.plot(x, ln_y)

plt.xlabel('FT Z Force')
plt.ylabel('Average Sensor Value')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()
