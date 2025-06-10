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
df = df[3359:12060].reset_index(drop = True)
df = df[(df['FT_Z'] < -0.03) | (df['FT_Z'] > 0.07)].reset_index(drop=True)

# Slice the data
ft_z = df['FT_Z'].to_numpy()
start_indices = np.where((ft_z[1:] > 0) & (ft_z[:-1] < 0))[0] + 1
end_indices = np.where((ft_z[1:] < 0) & (ft_z[:-1] > 0))[0] + 1

print(start_indices)
print(end_indices)

rounds = []
for i in range(11):
    if (i != 2):
        rounds.append((start_indices[i], end_indices[i]))

# Trim to 10 rounds and concatenate the data
round_data = []
for start, end in rounds:
    round_data.append(df.iloc[start:end])
df = pd.concat(round_data).reset_index(drop=True)

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
points_per_round = len(sensor_data) // 10
sensor_data = sensor_data[:points_per_round * 10]
data_10 = sensor_data.reshape(10, points_per_round)
calibrated_10 = cal_linear(data_10)

# Reshape the FT data
ft_data = df['FT_Z'].to_numpy()
ft_data = ft_data[:points_per_round * 10]
ft_10 = ft_data.reshape(10, points_per_round)

# Calculate sensor mean and std
sensor_mean = np.mean(calibrated_10, axis = 0)
sensor_std = np.std(calibrated_10, axis = 0)

# Calculate ft mean and std
ft_mean = np.mean(ft_10, axis = 0)
ft_std = np.std(ft_10, axis = 0)

# Plot
x = np.arange(len(sensor_mean))
plt.figure(figsize=(8, 6))
plt.plot(x, sensor_mean, label='Sensor Predict', color='blue')
plt.fill_between(x,
                 sensor_mean - sensor_std,
                 sensor_mean + sensor_std,
                 color='blue', alpha=0.2)

plt.plot(x, ft_mean, label='Ground Truth', color='orange')
plt.fill_between(x,
                 ft_mean - ft_std,
                 ft_mean + ft_std,
                 color='orange', alpha=0.2)

# # Plot all rounds
# plt.figure(figsize=(12, 6))
# for i in range(10):
#     plt.plot(ft_10[i], label=f'Round {i+1}', alpha=0.7)

plt.ylim(0, 4)
plt.xlim(0, 175)
plt.xlabel('Index')
plt.ylabel('Force (N)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()