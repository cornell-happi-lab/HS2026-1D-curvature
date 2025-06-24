import cv2
import numpy as np
import serial

# ---- Serial Setup ----
ser = serial.Serial('COM3', 115200, timeout=0.01)

# ---- Constants ----
GRID_ROWS = 4
GRID_COLS = 4
MAX_FORCE = 3.28  # in Newtons
MODEL_COEF = 0.009027
MODEL_INTERCEPT = -6.25

# ---- OpenCV Setup ----
WINDOW_NAME = "Real-Time Force Heatmap"
cv2.namedWindow(WINDOW_NAME)

# ---- Color Bar Setup ----
def create_colorbar(height, width=60):
    gradient = np.linspace(1, 0, height).reshape(height, 1)  # top = max, bottom = min
    gradient_img = np.tile(gradient, (1, width))  # make it wide
    colormap = cv2.applyColorMap((gradient_img * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Add labels for 0 and MAX_FORCE
    bar_img = colormap.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bar_img, f'{MAX_FORCE:.2f}N', (5, 20), font, 0.5, (255, 255, 255), 1)
    cv2.putText(bar_img, '0N', (5, height - 10), font, 0.5, (255, 255, 255), 1)
    return bar_img

def sensor_to_force(sensor_val):
    return MODEL_COEF * sensor_val + MODEL_INTERCEPT

def draw_heatmap(force_grid):
    norm = np.clip(force_grid / MAX_FORCE, 0, 1)
    heatmap = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return heatmap

while True:
    try:
        line = ser.readline().decode(errors='ignore').strip()
        if not line:
            continue

        # Parse 16 sensor values
        parts = line.split(',')
        if len(parts) != 16:
            continue

        sensor_vals = np.array([int(val) for val in parts])
        force_vals = sensor_to_force(sensor_vals).reshape(GRID_ROWS, GRID_COLS)

        heatmap = draw_heatmap(force_vals)
        heatmap = cv2.resize(heatmap, (400, 400), interpolation=cv2.INTER_NEAREST)

        colorbar = create_colorbar(height=400)

        combined = np.hstack((heatmap, colorbar))

        cv2.imshow(WINDOW_NAME, combined)

        # Press ESC to exit
        if cv2.waitKey(1) == 27:
            break

    except Exception as e:
        print("Error:", e)

ser.close()
cv2.destroyAllWindows()
