import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("calibration_data/dynamic_test_single_curved.csv")
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
print(baseline) # 677.9603944124897

df['Sensor'] = df['Sensor'] - baseline
df['Sensor'] = df['Sensor'].clip(lower = 0)

# Filter out the near-zero FT Z values
# df = df[3359:12060].reset_index(drop = True)
df = df[(df['FT_Z'] < -0.03) | (df['FT_Z'] > 0.07)].reset_index(drop=True)

# Calibration Function
def cal_linear(raw_data):
    force = 0.01508868863 * raw_data - 0.1933400104
    force = np.where(force < 0, 0, force)
    return moving_average(force, 10)

def cal_quad(raw_data):
    result = 6.07799109e-05 * (raw_data**2) +1.02982734e-02 * raw_data -1.38433643e-01
    result = np.where(raw_data < 164.3, 0, result)
    return result

def cal_power(raw_data):
    force = 0.0019086776537168572 * (raw_data**1.47) -0.14646366590244728
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
ax_sensor.set_ylim(0, 175)
# ax_sensor.plot(df.index, df['Sensor_smooth'], color = "Gray", label = "Raw Value")

ax_force = ax_sensor.twinx()
ax_force.set_label("Force (N)")
ax_force.set_ylim(0, 4)
ax_force.plot(df.index, df['FT_Z'], color = "Blue", label = "FT Z Force")
# ax_force.plot(df.index, cal_quad(df['Sensor']), color = "Black", label = "Quadratic Fit")
ax_force.plot(df.index, cal_linear(df['Sensor_smooth']), color = "green", label = "Linear Fit")
# ax_force.plot(df.index, cal_power(df['Sensor_smooth']), color = "purple", label = "Power Fit")
# ax_force.plot(df.index, cal_exp(df['Sensor']), color = "green", label = "Exp Fit")


plt.grid(True)
fig.tight_layout()
plt.legend()
plt.show()