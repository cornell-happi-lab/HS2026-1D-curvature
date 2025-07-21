import nidaqmx
import serial
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------- Arduino Thread ----------
arduino_value = None
lock = threading.Lock()

def read_arduino():
    global arduino_value
    arduino = serial.Serial('COM3', 115200, timeout=1)
    time.sleep(2)
    while True:
        try:
            line = arduino.readline().decode('ascii', errors='ignore').strip()
            with lock:
                arduino_value = line
        except Exception as e:
            print("Arduino error:", e)
        time.sleep(0.001)

# Start Arduino reader thread
arduino_thread = threading.Thread(target=read_arduino, daemon=True)
arduino_thread.start()

# ---------- NI DAQ and Main Loop ----------
calibration_matrix = np.array([
    [0.03133, 0.10222, 0.10361, -6.63382, 0.01609, 6.57641],
    [0.21162, 8.06214, 0.15262, -3.75490, -0.15622, -3.89770],
    [7.52422, -0.09766, 7.42244, 0.10983, 7.50850, 0.21699],
    [0.47851, 48.61235, 42.25686, -21.68818, -42.88027, -24.77775],
    [-48.77394, -0.16167, 23.21350, 40.69095, 25.01598, -39.08706],
    [0.69014, 30.86257, -0.40559, 28.61573, 0.04423, 29.42892],
])

bias = None
sensor_data = []
ft_data = []

print("Starting DAQ task...")
with nidaqmx.Task() as task:
    for ch in range(6):
        task.ai_channels.add_ai_voltage_chan(f"Dev1/ai{ch}")

    print("Collecting bias...")
    time.sleep(1.0)

    # Biasing
    bias_samples = []
    for _ in range(100):
        voltages = np.array(task.read())
        bias_samples.append(voltages)
        time.sleep(0.01)
    bias = np.mean(bias_samples, axis=0)
    print("Bias set to:", bias)

    print("Recording data. Press Ctrl+C to stop...")

    try:
        while True:
            voltages = np.array(task.read())
            corrected_voltages = voltages - bias
            ft = calibration_matrix @ corrected_voltages
            zforce = ft[2]

            with lock:
                sensor_str = arduino_value

            try:
                sensor_val = float(sensor_str)
                sensor_data.append(sensor_val)
                ft_data.append(zforce)
                print(f"Sensor: {sensor_val:.2f}, FT Z: {zforce:.2f}")
            except:
                print("Invalid sensor value:", sensor_str)

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nRecording stopped.")

# ---------- Save and Plot ----------
df = pd.DataFrame({
    'Sensor': sensor_data,
    'FT_Z': ft_data
})

# Optional: Save to CSV
df.to_csv("dynamic_test_2nd_sensor2.csv", index=False)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(df['Sensor'], df['FT_Z'], s=10, alpha=0.7)
plt.xlabel("Sensor Value (Arduino)")
plt.ylabel("Z Force (FT Sensor)")
plt.title("Sensor Value vs. FT Sensor Z Force")
plt.grid(True)
plt.tight_layout()
plt.show()
