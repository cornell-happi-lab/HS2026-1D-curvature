import serial
import numpy as np
import time
import csv

# === CONFIGURE SERIAL PORT ===
SERIAL_PORT = 'COM3'  # Change this to your actual port
BAUD_RATE = 115200

# === SETTINGS ===
NUM_SAMPLES = 100
VALUES_PER_SAMPLE = 16  # 4x4 grid flattened

# === STORAGE ===
all_values = []

# === OPEN SERIAL CONNECTION ===
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
time.sleep(2)  # Wait for Arduino to reset

print("Collecting data...")

while len(all_values) < NUM_SAMPLES:
    line = ser.readline().decode('utf-8', errors='replace').strip()
    if not line:
        continue
    try:
        values = [int(v) for v in line.split(',')]
        if len(values) == VALUES_PER_SAMPLE:
            print(values)
            all_values.append(values)
    except ValueError:
        continue  # Ignore malformed lines

ser.close()
print("Data collection complete.")

CSV_FILENAME = 'sensor4_sh_84.csv'

with open(CSV_FILENAME, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(all_values)

# === CONVERT TO NUMPY ARRAY ===
all_values_np = np.array(all_values)

# === CALCULATE MEAN & STDDEV ===
sensor_means = np.mean(all_values_np, axis = 0)
all_means = np.mean(sensor_means)
std_val = np.std(sensor_means)

# === DISPLAY RESULTS ===
print(sensor_means)
print(f"All Mean: {all_means}")
print(f"Standard Deviation: {std_val:.2f}")
