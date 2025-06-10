import nidaqmx
import serial
import threading
import time
import numpy as np
import pandas as pd
import keyboard  # pip install keyboard

# ---------- Shared Variables ----------
arduino_value = None
lock = threading.Lock()
latest_sensor = np.nan
latest_ft_z = np.nan
capture_now = False
step_data = []

# ---------- Arduino Thread ----------
def read_arduino():
    global arduino_value
    arduino = serial.Serial('COM3', 9600, timeout=1)
    time.sleep(2)
    while True:
        try:
            line = arduino.readline().decode('ascii', errors='ignore').strip()
            with lock:
                arduino_value = line
        except Exception as e:
            print("Arduino error:", e)
        time.sleep(0.001)

arduino_thread = threading.Thread(target=read_arduino, daemon=True)
arduino_thread.start()

# ---------- Main DAQ Thread ----------
def run_daq():
    global latest_sensor, latest_ft_z, step_data

    calibration_matrix = np.array([
        [0.03133, 0.10222, 0.10361, -6.63382, 0.01609, 6.57641],
        [0.21162, 8.06214, 0.15262, -3.75490, -0.15622, -3.89770],
        [7.52422, -0.09766, 7.42244, 0.10983, 7.50850, 0.21699],
        [0.47851, 48.61235, 42.25686, -21.68818, -42.88027, -24.77775],
        [-48.77394, -0.16167, 23.21350, 40.69095, 25.01598, -39.08706],
        [0.69014, 30.86257, -0.40559, 28.61573, 0.04423, 29.42892],
    ])

    print("Starting DAQ task...")
    with nidaqmx.Task() as task:
        for ch in range(6):
            task.ai_channels.add_ai_voltage_chan(f"Dev1/ai{ch}")

        # ---------- Bias ----------
        print("Collecting bias...")
        time.sleep(1.0)
        bias_samples = []
        for _ in range(100):
            voltages = np.array(task.read())
            bias_samples.append(voltages)
            time.sleep(0.01)
        bias = np.mean(bias_samples, axis=0)
        print("Bias set to:", bias)

        step = 1
        buffer_sensor = []
        buffer_ft = []

        print("\nPress ENTER to record a step (100 samples). Press ESC to stop.\n")

        while True:
            voltages = np.array(task.read())
            corrected_voltages = voltages - bias
            ft = calibration_matrix @ corrected_voltages
            zforce = ft[2]

            with lock:
                sensor_str = arduino_value

            try:
                sensor_val = float(sensor_str)
                latest_sensor = sensor_val
                latest_ft_z = zforce
                print(f"[LIVE] Sensor: {sensor_val:.2f}, FT Z: {zforce:.2f}")
            except:
                print(f"[LIVE] Invalid sensor: {sensor_str}")
                continue

            # Save samples if in capture mode
            if keyboard.is_pressed("enter"):
                print(f"\n[STEP {step}] Collecting 100 samples...")
                sensor_vals = []
                ft_vals = []
                for _ in range(100):
                    voltages = np.array(task.read())
                    corrected_voltages = voltages - bias
                    ft = calibration_matrix @ corrected_voltages
                    zforce = ft[2]

                    with lock:
                        sensor_str = arduino_value

                    try:
                        sensor_val = float(sensor_str)
                        sensor_vals.append(sensor_val)
                        ft_vals.append(zforce)
                        print(f"[Step {step}] Sensor: {sensor_val:.2f}, FT Z: {zforce:.2f}")
                    except:
                        print(f"[Step {step}] Invalid: {sensor_str}")
                        sensor_vals.append(np.nan)
                        ft_vals.append(np.nan)
                    time.sleep(0.01)

                df = pd.DataFrame({
                    'Sensor': sensor_vals,
                    'FT_Z': ft_vals,
                    'Step': step
                })
                step_data.append(df)
                print(f"[STEP {step}] Done.\n")
                step += 1

                # wait until ENTER released to avoid repeated steps
                while keyboard.is_pressed("enter"):
                    time.sleep(0.05)

            # Break if ESC is pressed
            if keyboard.is_pressed("esc"):
                print("\nESC pressed. Stopping...\n")
                break

            time.sleep(0.01)

# ---------- Start DAQ ----------
run_daq()

# ---------- Save to CSV ----------
if step_data:
    full_df = pd.concat(step_data, ignore_index=True)
    full_df.to_csv("static_test_2nd_sensor4.csv", index=False)
    print("Data saved to discrete_sensor_data.csv")
else:
    print("No step data collected.")
