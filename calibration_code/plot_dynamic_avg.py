import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("calibration_data/dynamic_test_single_flat4.csv")
# df = pd.read_csv("calibration_data/dynamic_test_single_curved.csv")
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

df['Sensor_smooth'] = moving_average(df['Sensor'], 5)

# ===== use your df, moving_average, cal_linear from above =====
ft = moving_average(df['FT_Z'].values, 7)                 # F/T (N)
sensor_force = cal_linear(df['Sensor_smooth'].values)      # calibrated sensor (N)

# ---- 1) Segment presses with hysteresis on F/T ----
thr_on, thr_off = 0.25, 0.15   # N
min_len = 80                   # samples
pre, post = 20, 20             # samples to include before/after

starts, ends = [], []
armed = True
for i in range(1, len(ft)):
    if armed and ft[i-1] <= thr_on and ft[i] > thr_on:
        starts.append(i)
        armed = False
    elif (not armed) and ft[i-1] >= thr_off and ft[i] < thr_off:
        ends.append(i)
        armed = True
if len(ends) < len(starts):
    starts = starts[:len(ends)]

# ---- 2) Build expanded windows around each press ----
win_ft, win_sensor, win_len = [], [], []
for s, e in zip(starts, ends):
    if e - s < min_len:
        continue
    s0 = max(0, s - pre)
    e0 = min(len(ft), e + post)
    win_ft.append(ft[s0:e0])
    win_sensor.append(sensor_force[s0:e0])
    win_len.append(e0 - s0)

if len(win_ft) == 0:
    raise RuntimeError("No valid presses found. Tune thr_on/thr_off/min_len.")

print(f"Found {len(win_ft)} presses (expanded by ±{pre}/{post} samples). Averaging…")

# ---- 3) Align at onset and pad to common length (NaN) ----
Lmax = max(win_len)
pad = lambda x, L: np.pad(x, (0, L - len(x)), constant_values=np.nan)

stack_ft = np.vstack([pad(x, Lmax) for x in win_ft])
stack_sensor = np.vstack([pad(x, Lmax) for x in win_sensor])

# sample axis: 0 corresponds to (onset - pre)
x = np.arange(Lmax) - pre + 20

mean_ft = np.nanmean(stack_ft, axis=0)
std_ft  = np.nanstd(stack_ft, axis=0)
mean_sensor = np.nanmean(stack_sensor, axis=0)
std_sensor  = np.nanstd(stack_sensor, axis=0)

# ---- 4) Plot: mean with std shading, different colors ----
plt.figure(figsize=(4, 4))
plt.plot(x, mean_sensor, color="tab:orange", label="Sensor Predict")
plt.fill_between(x, mean_sensor - std_sensor, mean_sensor + std_sensor,
                 color="tab:orange", alpha=0.25)

plt.plot(x, mean_ft, color="tab:blue", label="Ground Truth")
plt.fill_between(x, mean_ft - std_ft, mean_ft + std_ft,
                 color="tab:blue", alpha=0.2)
plt.xlabel("Time Stamp")
plt.ylabel("Force (N)")
plt.xlim(0, 160)
plt.ylim(0, 2.5)
# plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()