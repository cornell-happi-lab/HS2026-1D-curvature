import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV
df = pd.read_csv("dynamic_test.csv")

# # Calculate the bias value for the sensor
# bias_region = df[(df['FT_Z'] >= -0.03) & (df['FT_Z'] <= 0.03)]

# # Use IQR to filter out the outliers
# Q1 = bias_region['sensor_predict'].quantile(0.25)
# Q3 = bias_region['sensor_predict'].quantile(0.75)
# IQR = Q3 - Q1
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR
# IQR_filtered = bias_region[(bias_region['sensor_predict'] >= lower_bound) & (bias_region['sensor_predict'] <= upper_bound)]
# bias_value = IQR_filtered['sensor_predict'].mean()

# # Minus the bias value from my sensor
# df['sensor_predict'] = df['sensor_predict'] - bias_value

# Filter out near-zero FT Z values
df = df[(df['FT_Z'] < -0.03) | (df['FT_Z'] > 0.03)].reset_index(drop=True)

# # Compute direction change
# colors = ['blue']  # first point has no previous comparison
# for i in range(1, len(df)):
#     if df['FT_Z'][i] > df['FT_Z'][i - 1] + 0.7:
#         colors.append('red')  # Rising
#     elif df['FT_Z'][i] < df['FT_Z'][i - 1] - 0.7:
#         colors.append('blue')  # Falling
#     else:
#         colors.append('gray')  # same

# # Calibration  Exponential Function
# def cal(raw_value):
#     return (-7.1538 + 1.2671 * np.log(raw_value))**5

# Calibration Log Function
def cal(raw_value):
    return np.exp((raw_value - 655.9421) / 83.5465)

# # Calculate the average value across each rounds
# samples_per_round = len(df) // 10

# total_samples = 10 * samples_per_round
# ft_z = df['FT_Z'].values[:total_samples].reshape(10, samples_per_round)
# sensor = df['Sensor'].values[:total_samples].reshape(10, samples_per_round)

# sensor_predict = cal(sensor)
# sensor_predict_mean = np.mean(sensor_predict, axis = 0)
# sensor_predict_std = np.std(sensor_predict, axis = 0)

# ft_mean = np.mean(-ft_z, axis = 0)
# ft_std = np.std(ft_z, axis = 0)

# Moving average
def moving_average(x, window):
    return np.convolve(x, np.ones(window) / window, mode = 'same')
df['Sensor_predict_smooth'] = moving_average(cal(df['Sensor']), 10)

# fig, ax_sensor = plt.subplots(figsize = (8, 6))
# ax_sensor.plot(df.index, df['Sensor'], color = 'blue', label = 'Sensor_smooth')
# ax_sensor.set_ylim(300, 900)


fig, ax = plt.subplots(figsize = (8, 6))
ax.plot(df.index, -df['FT_Z'], color = 'blue')
# ax.plot(df.index, df['Sensor'], color = 'Orange')
ax.plot(df.index, moving_average(cal(df['Sensor']), 10), color = 'Green')

fig.tight_layout()
# plt.legend()
plt.grid(True)
plt.show()