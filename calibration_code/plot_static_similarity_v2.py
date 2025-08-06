import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

df = pd.read_csv("calibration_data/sensor5_cy_0.csv")
df['FT_Z'] = -df['FT_Z']
df['block'] = df['S5'] + df['S6'] + df['S9'] + df['S10']
grouped = df.groupby('Step')
avg_ft = grouped['FT_Z'].mean()
avg_sensor = grouped['block'].mean() 
std_sensor = grouped['block'].std()

df1 = pd.read_csv("calibration_data/sensor5_cy_20.csv")
df1['FT_Z'] = -df1['FT_Z']
df1['block'] = df1['S5'] + df1['S6'] + df1['S9'] + df1['S10']
grouped1 = df1.groupby('Step')
avg_ft1 = grouped1['FT_Z'].mean()
avg_sensor1 = grouped1['block'].mean() 
std_sensor1 = grouped1['block'].std()

df2 = pd.read_csv("calibration_data/sensor5_cy_40.csv")
df2['FT_Z'] = -df2['FT_Z']
df2['block'] = df2['S5'] + df2['S6'] + df2['S9'] + df2['S10']
grouped2 = df2.groupby('Step')
avg_ft2 = grouped2['FT_Z'].mean()
avg_sensor2 = grouped2['block'].mean() 
std_sensor2 = grouped2['block'].std()

df3 = pd.read_csv("calibration_data/sensor5_cy_60.csv")
df3['FT_Z'] = -df3['FT_Z']
df3['block'] = df3['S5'] + df3['S6'] + df3['S9'] + df3['S10']
grouped3 = df3.groupby('Step')
avg_ft3 = grouped3['FT_Z'].mean()
avg_sensor3 = grouped3['block'].mean() 
std_sensor3 = grouped3['block'].std()

df4 = pd.read_csv("calibration_data/sensor5_cy_80.csv")
df4['FT_Z'] = -df4['FT_Z']
df4['block'] = df4['S5'] + df4['S6'] + df4['S9'] + df4['S10']
grouped4 = df4.groupby('Step')
avg_ft4 = grouped4['FT_Z'].mean()
avg_sensor4 = grouped4['block'].mean() 
std_sensor4 = grouped4['block'].std()

df5 = pd.read_csv("calibration_data/sensor5_cy_100.csv")
df5['FT_Z'] = -df5['FT_Z']
df5['block'] = df5['S5'] + df5['S6'] + df5['S9'] + df5['S10']
grouped5 = df5.groupby('Step')
avg_ft5 = grouped5['FT_Z'].mean()
avg_sensor5 = grouped5['block'].mean() 
std_sensor5 = grouped5['block'].std()


plt.errorbar(
    x=avg_ft,
    y=avg_sensor,
    yerr=std_sensor,
    fmt='o-',
    capsize=5,
    ecolor='gray',
    elinewidth=1,
    markersize=4,
    color = "black",
    label = "0"
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
    color = "brown",
    label = "20"
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
    color = "grey",
    label = "40"
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
    color = "lightblue",
    label = "60"
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
    color = "lightgreen",
    label = "80"
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
    color = "orchid",
    label = "100"
)

plt.xlabel('FT Z Force')
plt.ylabel('Sensor Value')
plt.grid(True)
plt.xlim(0, 20)
plt.legend()
plt.tight_layout()
plt.show()
    