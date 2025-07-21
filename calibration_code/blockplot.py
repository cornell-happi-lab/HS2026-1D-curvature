import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


df = pd.read_csv("calibration_data/block1_3.csv")
df['block'] = df['S2'] + df['S3'] + df['S6'] + df['S7']
df['FT_Z'] = -df['FT_Z']
grouped = df.groupby('Step')
avg_ft = grouped['FT_Z'].mean()
avg_sensor = grouped['block'].mean() 
std_sensor = grouped['block'].std()

df1 = pd.read_csv("calibration_data/block1_2.csv")
df1['block'] = df1['S1'] + df1['S2'] + df1['S5'] + df1['S6']
df1['FT_Z'] = -df1['FT_Z']
grouped1 = df1.groupby('Step')
avg_ft1 = grouped1['FT_Z'].mean()
avg_sensor1 = grouped1['block'].mean() 
std_sensor1 = grouped1['block'].std()

df2 = pd.read_csv("calibration_data/block1_1.csv")
df2['block'] = df2['S0'] + df2['S1'] + df2['S4'] + df2['S5']
df2['FT_Z'] = -df2['FT_Z']
grouped2 = df2.groupby('Step')
avg_ft2 = grouped2['FT_Z'].mean()
avg_sensor2 = grouped2['block'].mean() 
std_sensor2 = grouped2['block'].std()

df3 = pd.read_csv("calibration_data/block2_1.csv")
df3['block'] = df3['S4'] + df3['S5'] + df3['S8'] + df3['S9']
df3['FT_Z'] = -df3['FT_Z']
grouped3 = df3.groupby('Step')
avg_ft3 = grouped3['FT_Z'].mean()
avg_sensor3 = grouped3['block'].mean() 
std_sensor3 = grouped3['block'].std()

df4 = pd.read_csv("calibration_data/block2_2.csv")
df4['block'] = df4['S5'] + df4['S6'] + df4['S9'] + df4['S10']
df4['FT_Z'] = -df4['FT_Z']
grouped4 = df4.groupby('Step')
avg_ft4 = grouped4['FT_Z'].mean()
avg_sensor4 = grouped4['block'].mean() 
std_sensor4 = grouped4['block'].std()

df5 = pd.read_csv("calibration_data/block2_3.csv")
df5['block'] = df5['S6'] + df5['S7'] + df5['S10'] + df5['S11']
df5['FT_Z'] = -df5['FT_Z']
grouped5 = df5.groupby('Step')
avg_ft5 = grouped5['FT_Z'].mean()
avg_sensor5 = grouped5['block'].mean() 
std_sensor5 = grouped5['block'].std()

df6 = pd.read_csv("calibration_data/block3_2.csv")
df6['block'] = df6['S9'] + df6['S10'] + df6['S13'] + df6['S14']
df6['FT_Z'] = -df6['FT_Z']
grouped6 = df6.groupby('Step')
avg_ft6 = grouped6['FT_Z'].mean()
avg_sensor6 = grouped6['block'].mean() 
std_sensor6 = grouped6['block'].std()

df7 = pd.read_csv("calibration_data/block3_3.csv")
df7['block'] = df7['S10'] + df7['S11'] + df7['S14'] + df7['S15']
df7['FT_Z'] = -df7['FT_Z']
grouped7 = df7.groupby('Step')
avg_ft7 = grouped7['FT_Z'].mean()
avg_sensor7 = grouped7['block'].mean() 
std_sensor7 = grouped7['block'].std()

df8 = pd.read_csv("calibration_data/block3_1.csv")
df8['block'] = df8['S8'] + df8['S9'] + df8['S12'] + df8['S13']
df8['FT_Z'] = -df8['FT_Z']
grouped8 = df8.groupby('Step')
avg_ft8 = grouped8['FT_Z'].mean()
avg_sensor8 = grouped8['block'].mean() 
std_sensor8 = grouped8['block'].std()

avg_fts = [avg_ft1, avg_ft2, avg_ft3, avg_ft4, 
           avg_ft5, avg_ft6, avg_ft7, avg_ft8,]
avg_sensors = [avg_sensor1, avg_sensor2, avg_sensor3, avg_sensor4, 
               avg_sensor5, avg_sensor6, avg_sensor7, avg_sensor8]

aligned_fts = []
aligned_sensors = []

for ft_series, sensor_series in zip(avg_fts, avg_sensors):
    ft_aligned, sensor_aligned = ft_series.align(sensor_series, join='inner')
    aligned_fts.append(ft_aligned)
    aligned_sensors.append(sensor_aligned)

# Step 3: Combine into one long series
all_ft_trimmed = pd.concat(aligned_fts)
all_sensor_trimmed = pd.concat(aligned_sensors)

# Filter to keep only force < 10 N
mask = all_ft_trimmed < 10
all_ft_trimmed = all_ft_trimmed[mask]
all_sensor_trimmed = all_sensor_trimmed[mask]

# Step 4: Fit linear model
def linear_model(x, a, b):
    return a * x + b

params, _ = curve_fit(linear_model, all_ft_trimmed, all_sensor_trimmed)
a, b = params
print(f"a: {a:.2f}, b: {b:.2f}")

# Step 5: Plot
plt.figure(figsize=(8, 6))

x_fit = np.linspace(0, 10, 100)
y_fit = linear_model(x_fit, a, b)


plt.errorbar(
    x=avg_ft,
    y=avg_sensor,
    yerr=std_sensor,
    fmt='o-',
    capsize=5,
    ecolor='gray',
    elinewidth=1,
    markersize=4,
    color = "black"
)

plt.errorbar(
    x=avg_ft1,
    y=avg_sensor1,
    yerr=std_sensor1,
    fmt='o-',
    capsize=5,
    ecolor='gray',
    elinewidth=1,
    markersize=4,
    color = "black"
)

plt.errorbar(
    x=avg_ft2,
    y=avg_sensor2,
    yerr=std_sensor2,
    fmt='o-',
    capsize=5,
    ecolor='gray',
    elinewidth=1,
    markersize=4,
    color = "black"
)

plt.errorbar(
    x=avg_ft3,
    y=avg_sensor3,
    yerr=std_sensor3,
    fmt='o-',
    capsize=5,
    ecolor='gray',
    elinewidth=1,
    markersize=4,
    color = "black"
)

plt.errorbar(
    x=avg_ft4,
    y=avg_sensor4,
    yerr=std_sensor4,
    fmt='o-',
    capsize=5,
    ecolor='gray',
    elinewidth=1,
    markersize=4,
    color = "black"
)

plt.errorbar(
    x=avg_ft5,
    y=avg_sensor5,
    yerr=std_sensor5,
    fmt='o-',
    capsize=5,
    ecolor='gray',
    elinewidth=1,
    markersize=4,
    color = "black"
)

plt.errorbar(
    x=avg_ft6,
    y=avg_sensor6,
    yerr=std_sensor6,
    fmt='o-',
    capsize=5,
    ecolor='gray',
    elinewidth=1,
    markersize=4,
    color = "black"
)

plt.errorbar(
    x=avg_ft7,
    y=avg_sensor7,
    yerr=std_sensor7,
    fmt='o-',
    capsize=5,
    ecolor='gray',
    elinewidth=1,
    markersize=4,
    color = "black"
)

plt.errorbar(
    x=avg_ft8,
    y=avg_sensor8,
    yerr=std_sensor8,
    fmt='o-',
    capsize=5,
    ecolor='gray',
    elinewidth=1,
    markersize=4,
    color = "black"
)

# Plot the fitted line
plt.plot(x_fit, y_fit, 'r', label=f'Fit: y = {a:.2f}x + {b:.2f}', linewidth=3)
plt.legend()

plt.xlabel('FT Z Force')
plt.ylabel('Sensor Value')
plt.grid(True)
# plt.legend()
plt.tight_layout()
# plt.ylim(0, 100)
# plt.xlim(0, 10)
plt.show()