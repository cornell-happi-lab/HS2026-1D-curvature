import nidaqmx
import serial
import threading
import time
import numpy as np

# ---------- Arduino Thread ----------
arduino_value = None  # Shared value
lock = threading.Lock()

def read_arduino():
    global arduino_value
    arduino = serial.Serial('COM4', 9600, timeout=1)
    time.sleep(2)  # Give Arduino time to boot

    while True:
        try:
            line = arduino.readline().decode('ascii', errors='ignore').strip()
            with lock:
                arduino_value = line
        except Exception as e:
            print("Arduino error:", e)
        time.sleep(0.001)  # Reduce CPU usage

# Start Arduino reader in separate thread
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

    # Main loop
    while True:
        try:
            voltages = np.array(task.read())
            corrected_voltages = voltages - bias
            ft = calibration_matrix @ corrected_voltages
            zforce = ft[2]

            with lock:
                sensor = arduino_value

            print(f"Sensor: {sensor}, FT Z: {zforce:.2f}")
            time.sleep(0.01)

        except Exception as e:
            print("Main loop error:", e)
