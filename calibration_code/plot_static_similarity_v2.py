import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# S0
df0 = pd.read_csv("calibration_data/static_test_node0.csv")
df0['FT_Z'] = -df0['FT_Z']
grouped0 = df0.groupby('Step')
avg_ft0 = grouped0['FT_Z'].mean()
avg_sensor0 = grouped0['S0'].mean() 
std_sensor0 = grouped0['S0'].std()

# S1
df1 = pd.read_csv("calibration_data/static_test_node1.csv")
df1['FT_Z'] = -df1['FT_Z']
grouped1 = df1.groupby('Step')
avg_ft1 = grouped1['FT_Z'].mean()
avg_sensor1 = grouped1['S1'].mean() 
std_sensor1 = grouped1['S1'].std()

# S2
df2 = pd.read_csv("calibration_data/static_test_node2.csv")
df2['FT_Z'] = -df2['FT_Z']
grouped2 = df2.groupby('Step')
avg_ft2 = grouped2['FT_Z'].mean()
avg_sensor2 = grouped2['S2'].mean() 
std_sensor2 = grouped2['S2'].std()

# S3
df3 = pd.read_csv("calibration_data/static_test_node3.csv")
df3['FT_Z'] = -df3['FT_Z']
grouped3 = df3.groupby('Step')
avg_ft3 = grouped3['FT_Z'].mean()
avg_sensor3 = grouped3['S3'].mean() 
std_sensor3 = grouped3['S3'].std()

# S4
df4 = pd.read_csv("calibration_data/static_test_node4.csv")
df4['FT_Z'] = -df4['FT_Z']
grouped4 = df4.groupby('Step')
avg_ft4 = grouped4['FT_Z'].mean()
avg_sensor4 = grouped4['S4'].mean() 
std_sensor4 = grouped4['S4'].std()

# S5
df5 = pd.read_csv("calibration_data/static_test_node5.csv")
df5['FT_Z'] = -df5['FT_Z']
grouped5 = df5.groupby('Step')
avg_ft5 = grouped5['FT_Z'].mean()
avg_sensor5 = grouped5['S5'].mean() 
std_sensor5 = grouped5['S5'].std()

# S6
df6 = pd.read_csv("calibration_data/static_test_node6.csv")
df6['FT_Z'] = -df6['FT_Z']
grouped6 = df6.groupby('Step')
avg_ft6 = grouped6['FT_Z'].mean()
avg_sensor6 = grouped6['S6'].mean() 
std_sensor6 = grouped6['S6'].std()

# S7
df7 = pd.read_csv("calibration_data/static_test_node7.csv")
df7['FT_Z'] = -df7['FT_Z']
grouped7 = df7.groupby('Step')
avg_ft7 = grouped7['FT_Z'].mean()
avg_sensor7 = grouped7['S7'].mean() 
std_sensor7 = grouped7['S7'].std()

# S8
df8 = pd.read_csv("calibration_data/static_test_node8.csv")
df8['FT_Z'] = -df8['FT_Z']
grouped8 = df8.groupby('Step')
avg_ft8 = grouped8['FT_Z'].mean()
avg_sensor8 = grouped8['S8'].mean() 
std_sensor8 = grouped8['S8'].std()

# S9
df9 = pd.read_csv("calibration_data/static_test_node9.csv")
df9['FT_Z'] = -df9['FT_Z']
grouped9 = df9.groupby('Step')
avg_ft9 = grouped9['FT_Z'].mean()
avg_sensor9 = grouped9['S9'].mean() 
std_sensor9 = grouped9['S9'].std()

# S10
df10 = pd.read_csv("calibration_data/static_test_node10.csv")
df10['FT_Z'] = -df10['FT_Z']
grouped10 = df10.groupby('Step')
avg_ft10 = grouped10['FT_Z'].mean()
avg_sensor10 = grouped10['S10'].mean() 
std_sensor10 = grouped10['S10'].std()

# S11
df11 = pd.read_csv("calibration_data/static_test_node11.csv")
df11['FT_Z'] = -df11['FT_Z']
grouped11 = df11.groupby('Step')
avg_ft11 = grouped11['FT_Z'].mean()
avg_sensor11 = grouped11['S11'].mean() 
std_sensor11 = grouped11['S11'].std()

# S12
df12 = pd.read_csv("calibration_data/static_test_node12.csv")
df12['FT_Z'] = -df12['FT_Z']
grouped12 = df12.groupby('Step')
avg_ft12 = grouped12['FT_Z'].mean()
avg_sensor12 = grouped12['S12'].mean() 
std_sensor12 = grouped12['S12'].std()

# S13
df13 = pd.read_csv("calibration_data/static_test_node13.csv")
df13['FT_Z'] = -df13['FT_Z']
grouped13 = df13.groupby('Step')
avg_ft13 = grouped13['FT_Z'].mean()
avg_sensor13 = grouped13['S13'].mean() 
std_sensor13 = grouped13['S13'].std()

# S14
df14 = pd.read_csv("calibration_data/static_test_node14.csv")
df14['FT_Z'] = -df14['FT_Z']
grouped14 = df14.groupby('Step')
avg_ft14 = grouped14['FT_Z'].mean()
avg_sensor14 = grouped14['S14'].mean() 
std_sensor14 = grouped14['S14'].std()

# S15
df15 = pd.read_csv("calibration_data/static_test_node15.csv")
df15['FT_Z'] = -df15['FT_Z']
grouped15 = df15.groupby('Step')
avg_ft15 = grouped15['FT_Z'].mean()
avg_sensor15 = grouped15['S15'].mean() 
std_sensor15 = grouped15['S15'].std()

# Step 1–2: Compute FT_Z threshold where sensor hits 1023
ft_thresholds = []

avg_fts = [avg_ft0, avg_ft1, avg_ft2, avg_ft4, 
           avg_ft7, avg_ft8, avg_ft10, avg_ft13]
avg_sensors = [avg_sensor0, avg_sensor1, avg_sensor2, avg_sensor4, 
               avg_sensor7, avg_sensor8, avg_sensor10, avg_sensor13]

# Store trimmed data
trimmed_ft = []
trimmed_sensor = []

for ft_series, sensor_series in zip(avg_fts, avg_sensors):
    # Align indices
    ft_series, sensor_series = ft_series.align(sensor_series, join='inner')
    
    # Find first FT_Z value where sensor hits 1023
    saturated = sensor_series[sensor_series >= 1023]
    if not saturated.empty:
        ft_limit = ft_series[saturated.index[0]]
    else:
        ft_limit = ft_series.max()  # If never saturates
    ft_thresholds.append(ft_limit)
    
# Step 2: Compute shared FT threshold
shared_threshold = np.mean(ft_thresholds)
print(f"Shared FT_Z threshold: {shared_threshold:.2f} N")

# Step 3: Trim all data to shared threshold
trimmed_ft = []
trimmed_sensor = []

for ft_series, sensor_series in zip(avg_fts, avg_sensors):
    ft_series, sensor_series = ft_series.align(sensor_series, join='inner')
    mask = ft_series <= shared_threshold
    trimmed_ft.append(ft_series[mask])
    trimmed_sensor.append(sensor_series[mask])

# Step 4: Combine for fitting
all_ft_trimmed = pd.concat(trimmed_ft)
all_sensor_trimmed = pd.concat(trimmed_sensor)

# Step 5: Fit line
def linear_model(x, a, b):
    return a * x + b

params, _ = curve_fit(linear_model, all_ft_trimmed, all_sensor_trimmed)
a, b = params

print(f"a: {a:.2f}, b: {b:.2f}")

# Step 6: Plot best-fit line
x_fit = np.linspace(all_ft_trimmed.min(), shared_threshold, 100)
y_fit = linear_model(x_fit, a, b)

# Plotting
plt.figure(figsize=(8, 6))

# Plot the fitted line
plt.plot(x_fit, y_fit, 'r--', label=f'Fit: y = {a:.2f}x + {b:.2f}', linewidth=2)
plt.legend()

plt.errorbar(
    x=avg_ft0,
    y=avg_sensor0,
    yerr=std_sensor0,
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
    color = "gray"
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
    color = "grey"
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
    color = "grey"
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

plt.errorbar(
    x=avg_ft9,
    y=avg_sensor9,
    yerr=std_sensor9,
    fmt='o-',
    capsize=5,
    ecolor='gray',
    elinewidth=1,
    markersize=4,
    color = "gray"
)

plt.errorbar(
    x=avg_ft10,
    y=avg_sensor10,
    yerr=std_sensor10,
    fmt='o-',
    capsize=5,
    ecolor='gray',
    elinewidth=1,
    markersize=4,
    color = "black"
)

plt.errorbar(
    x=avg_ft11,
    y=avg_sensor11,
    yerr=std_sensor11,
    fmt='o-',
    capsize=5,
    ecolor='gray',
    elinewidth=1,
    markersize=4,
    color = "gray"
)

plt.errorbar(
    x=avg_ft12,
    y=avg_sensor12,
    yerr=std_sensor12,
    fmt='o-',
    capsize=5,
    ecolor='gray',
    elinewidth=1,
    markersize=4,
    color = "gray"
)

plt.errorbar(
    x=avg_ft13,
    y=avg_sensor13,
    yerr=std_sensor13,
    fmt='o-',
    capsize=5,
    ecolor='gray',
    elinewidth=1,
    markersize=4,
    color = "black"
)

plt.errorbar(
    x=avg_ft14,
    y=avg_sensor14,
    yerr=std_sensor14,
    fmt='o-',
    capsize=5,
    ecolor='gray',
    elinewidth=1,
    markersize=4,
    color = "grey"
)

plt.errorbar(
    x=avg_ft15,
    y=avg_sensor15,
    yerr=std_sensor15,
    fmt='o-',
    capsize=5,
    ecolor='gray',
    elinewidth=1,
    markersize=4,
    color = "gray"
)

plt.xlabel('FT Z Force')
plt.ylabel('Sensor Value')
plt.grid(True)
# plt.legend()
plt.tight_layout()
# plt.ylim(0, 100)
# plt.xlim(0, 10)
plt.show()
