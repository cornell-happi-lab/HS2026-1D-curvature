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
df['Sensor'] = df['Sensor'].clip(lower = 0)

# Filter out the near-zero FT Z values
df = df[(df['FT_Z'] < -0.3) | (df['FT_Z'] > 0.3)].reset_index(drop=True)

# Calibration Function
def cal_linear(raw_data):
    force = 0.01395828 * raw_data + 0.0677103
    force = np.where(force < 0, 0, force)
    # return moving_average(force, 10)
    return force

# Moving average filter
def moving_average(x, window):
    return np.convolve(x, np.ones(window) / window, mode = 'same')

# Reshape the sensor data
sensor_data = df['Sensor'].to_numpy()
points_per_round = len(sensor_data) // 16
sensor_data = sensor_data[:points_per_round * 16]
data_16 = sensor_data.reshape((16, points_per_round))
calibrated_16 = cal_linear(data_16)

# Reshape the FT data
ft_data = df['FT_Z'].to_numpy()
ft_data = ft_data[:points_per_round * 16]
ft_16 = ft_data.reshape(16, points_per_round)

# Calculate sensor mean and std
sensor_mean = np.mean(calibrated_16, axis = 0)
sensor_std = np.std(calibrated_16, axis = 0)

# Calculate ft mean and std
ft_mean = np.mean(ft_16, axis = 0)
ft_std = np.std(ft_16, axis = 0)

# Plot
x = np.arange(len(sensor_mean))
plt.figure(figsize=(8, 6))
plt.plot(x, sensor_mean, label='Average Predicted Force', color='blue')
plt.fill_between(x,
                 sensor_mean - sensor_std,
                 sensor_mean + sensor_std,
                 color='blue', alpha=0.2)

plt.plot(x, ft_mean, label='F/T Force', color='orange')
plt.fill_between(x,
                 ft_mean - ft_std,
                 ft_mean + ft_std,
                 color='orange', alpha=0.2)

# Plot all rounds
plt.figure(figsize=(12, 6))
for i in range(16):
    plt.plot(ft_16[i], label=f'Round {i+1}', alpha=0.7)

plt.xlabel('Index')
plt.ylabel('Force (N)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()