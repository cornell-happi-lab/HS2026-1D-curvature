import cv2
import numpy as np
import serial
import time

# ---- Serial Setup ----
ser = serial.Serial('COM3', 115200, timeout=0.01)

# ---- Constants ----
GRID_ROWS = 4
GRID_COLS = 4
MAX_FORCE = 10  # in Newtons
MODEL_COEF = 0.01076889942
MODEL_INTERCEPT = -0.2713762653
BIAS_SAMPLES = 100

# ---- OpenCV Setup ----
WINDOW_NAME = "Real-Time Force Heatmap"
cv2.namedWindow(WINDOW_NAME)

# ---- Color Bar Setup ----
def create_colorbar(height, width=60):
    gradient = np.linspace(1, 0, height).reshape(height, 1)
    gradient_img = np.tile(gradient, (1, width))
    colormap = cv2.applyColorMap((gradient_img * 255).astype(np.uint8), cv2.COLORMAP_JET)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(colormap, f'{MAX_FORCE:.2f}N', (5, 20), font, 0.5, (255, 255, 255), 1)
    cv2.putText(colormap, '0N', (5, height - 10), font, 0.5, (255, 255, 255), 1)
    return colormap

# ---- Transform 4x4 to 3x3 using 2x2 sum blocks ----
def compute_2x2_sums(sensor_4x4):
    sensor_3x3 = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            block = sensor_4x4[i:i+2, j:j+2]
            sensor_3x3[i, j] = np.sum(block)
    return sensor_3x3

# ---- Sensor to Force Mapping ----
def sensor_to_force(sensor_val):
    return MODEL_COEF * sensor_val + MODEL_INTERCEPT

# ---- Heatmap Drawing ----
def draw_heatmap(force_grid):
    norm = np.clip(force_grid / MAX_FORCE, 0, 1)
    heatmap = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return heatmap

# ---- Bias Collection ----
print("Collecting bias... Please keep sensor untouched.")
bias_data = []

while len(bias_data) < BIAS_SAMPLES:
    try:
        line = ser.readline().decode(errors='ignore').strip()
        parts = line.split(',')
        if len(parts) != 16:
            continue
        values = np.array([int(val) for val in parts])
        bias_data.append(values)
    except:
        continue

bias_array = np.mean(bias_data, axis=0)
print("Bias set to:", bias_array)

# ---- Main Loop ----
while True:
    try:
        line = ser.readline().decode(errors='ignore').strip()
        if not line:
            continue

        parts = line.split(',')
        if len(parts) != 16:
            continue

        raw_vals = np.array([int(val) for val in parts])
        corrected_vals = raw_vals - bias_array
        corrected_vals = np.clip(corrected_vals, 0, None)  # avoid negative values

        sensor_4x4 = corrected_vals.reshape(4, 4)

        sensor_3x3 = compute_2x2_sums(sensor_4x4)
        force_3x3 = sensor_to_force(sensor_3x3)

        heatmap = draw_heatmap(force_3x3)
        heatmap = cv2.resize(heatmap, (400, 400), interpolation=cv2.INTER_NEAREST)

        colorbar = create_colorbar(height=400)
        combined = np.hstack((heatmap, colorbar))

        cv2.imshow(WINDOW_NAME, combined)

        if cv2.waitKey(1) == 27:  # ESC
            break

    except Exception as e:
        print("Error:", e)

ser.close()
cv2.destroyAllWindows()
