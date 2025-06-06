import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv("static_test.csv")
df1['FT_Z'] = -df1['FT_Z']
grouped1 = df1.groupby('Step')
ft1 = grouped1['FT_Z'].mean()
sensor1_avg = grouped1['Sensor'].mean()
sensor1_avg = sensor1_avg - sensor1_avg.iloc[0]
# print(sensor1_avg)
sensor1_std = grouped1['Sensor'].std()

df2 = pd.read_csv("static_test_2nd_sensor.csv")
df2['FT_Z'] = -df2['FT_Z']
grouped2 = df2.groupby('Step')
ft2 = grouped2['FT_Z'].mean()
sensor2_avg = grouped2['Sensor'].mean()
sensor2_avg = sensor2_avg - sensor2_avg.iloc[0]
sensor2_std = grouped2['Sensor'].std()

# Min-Max Normalization
sensor1_norm = (sensor1_avg - sensor1_avg.min()) / (sensor1_avg.max() - sensor1_avg.min())
sensor2_norm = (sensor2_avg - sensor2_avg.min()) / (sensor2_avg.max() - sensor2_avg.min())

plt.figure(figsize = (8, 6))
plt.scatter(ft1, sensor1_norm, color = "Blue", label = "Sensor1")
plt.scatter(ft2, sensor2_norm, color = "Orange", label = "Sensor2")
# plt.errorbar(ft1, sensor1_avg, yerr = sensor1_std, fmt = 'o', markersize = 4, capsize = 5, elinewidth = 1, ecolor = 'gray', label = "Sensor1")
# plt.errorbar(ft2, sensor2_avg, yerr = sensor2_std, fmt = 'o', markersize = 4, capsize = 5, elinewidth = 1, ecolor = 'gray', label = "Sensor2")

plt.xlim(0, 17)
plt.ylim(0, 1)
plt.xlabel("FT Z Force")
plt.ylabel('Average Sensor Value')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()