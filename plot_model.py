import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("dynamic_test_2N_2nd_sensor.csv")
df['FT_Z'] = -df['FT_Z']

# Find the baseline for the sensor and minus it
baseline_region = df[(df['FT_Z'] >= -0.03) & (df['FT_Z'] <= 0.03)]
Q1 = baseline_region['Sensor'].quantile(0.25)
Q3 = baseline_region['Sensor'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
IQR_filtered = baseline_region[(baseline_region['Sensor'] >= lower_bound) & (baseline_region['Sensor'] <= upper_bound)]
baseline = IQR_filtered['Sensor'].mean()
print(baseline)

df['Sensor'] = df['Sensor'] - baseline
print(baseline)
df['Sensor'] = df['Sensor'].clip(lower = 0)

# Filter out the near-zero FT Z values
df = df[(df['FT_Z'] < -0.03) | (df['FT_Z'] > 0.03)].reset_index(drop=True)

# Calibration Function
def cal_linear(raw_data):
    force = 0.0093842 * raw_data -1.54898488
    force = np.where(force < 0, 0, force)
    return moving_average(force, 10)

def cal_quad(raw_data):
    result = 3.42089499e-05 * (raw_data**2) -8.71522057e-03 * raw_data +4.62796956e-01
    result = np.where(raw_data < 164.3, 0, result)
    return result

def cal_power(raw_data):
    force = 0.0004731482094181381 * (raw_data**1.47) -0.6678598570322021
    force = np.where(force < 0, 0, force)
    return moving_average(force, 10)

def cal_exp(raw_data):
    ln_force = -3.1994982717702247 + 0.009745754189733688 * raw_data
    return np.exp(ln_force)

# Moving average filter
def moving_average(x, window):
    return np.convolve(x, np.ones(window) / window, mode = 'same')

df['Sensor_smooth'] = moving_average(df['Sensor'], 10)

fig, ax_sensor = plt.subplots(figsize = (8, 6))
ax_sensor.set_label("Sensor")
ax_sensor.set_ylim(0, 700)
ax_sensor.plot(df.index, df['Sensor'], color = "Gray")

ax_force = ax_sensor.twinx()
ax_force.set_label("Force (N)")
ax_force.set_ylim(0, 5)
ax_force.plot(df.index, df['FT_Z'], color = "Blue")
# ax_force.plot(df.index, cal_quad(df['Sensor']), color = "Green", label = "Quadratic Fit")
# ax_force.plot(df.index, cal_linear(df['Sensor']), color = "green", label = "Linear Fit")
# ax_force.plot(df.index, cal_power(df['Sensor']), color = "green", label = "Power Fit")
ax_force.plot(df.index, moving_average(cal_exp(df['Sensor_smooth']), 10), color = "green", label = "Exp Fit")

plt.grid(True)
fig.tight_layout()
plt.show()  