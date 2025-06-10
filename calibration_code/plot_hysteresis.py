import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("calibration_data/dynamic_test_2nd_sensor3.csv")
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
df = df[3359:12060].reset_index(drop = True)
df = df[(df['FT_Z'] < -0.03) | (df['FT_Z'] > 0.03)].reset_index(drop=True)

# Calibration Function
def cal_linear(raw_data):
    force = 0.01395828 * raw_data + 0.0677103
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


# Plot
plt.figure(figsize=(8, 6))
plt.scatter(df['FT_Z'], cal_linear(df['Sensor']), s = 7)

plt.xlim(0, 4)
plt.ylim(0, 4)
plt.xlabel('Ground Truth (N)')
plt.ylabel('Sensor Predict (N)')
plt.grid(True)
plt.tight_layout()
plt.show()