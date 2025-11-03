# record_5s_after_enter.py
import time, math, threading
from collections import deque
from datetime import datetime
import numpy as np
import pandas as pd

# ================= User Config =================
COM_PORT          = "COM5"      # Arduino port
BAUD              = 115200
NI_DEVICE         = "Dev1"      # NI-DAQ device name (e.g., Dev1)
SAMPLE_HZ         = 1000        # FT sample rate
RECORD_SEC        = 5.0         # 5 seconds after Enter
SENSOR_BIAS_SEC   = 2.0         # how long to gather sensor bias (starts immediately)
FT_BIAS_SEC       = 1.0         # how long to gather FT bias (starts immediately)
CURV              = 25.3        # curvature used in "force_curv"
SIMULATION        = False       # set True to test without NI-DAQ

# =============== ATI Calibration ===============
CALIBRATION_MATRIX = np.array([
    [0.03133, 0.10222, 0.10361, -6.63382, 0.01609, 6.57641],
    [0.21162, 8.06214, 0.15262, -3.75490, -0.15622, -3.89770],
    [7.52422, -0.09766, 7.42244, 0.10983, 7.50850, 0.21699],
    [0.47851, 48.61235, 42.25686, -21.68818, -42.88027, -24.77775],
    [-48.77394, -0.16167, 23.21350, 40.69095, 25.01598, -39.08706],
    [0.69014, 30.86257, -0.40559, 28.61573, 0.04423, 29.42892],
])

# ====== Sensor calibration (your polynomial) ======
def coeffs_from_curvature(c: float):
    # F = a3*S^3 + a2*S^2 + a1*S + a0, S = (sensor_raw - bias)
    a3 = 0.0
    a2 = -0.000014
    a1 = 0.009625 + (-0.000372 * c) + (0.0000050 * (c**2))
    a0 = 0.0
    scale = 4.0  # four summed blocks
    return np.array([scale*a3, scale*a2, scale*a1, scale*a0], dtype=float)

COEFFS_CURV = coeffs_from_curvature(CURV)
COEFFS_C0   = coeffs_from_curvature(0.0)

def poly_eval(coeffs, x):
    return ((coeffs[0]*x + coeffs[1])*x + coeffs[2])*x + coeffs[3]

# =============== Hardware availability ===============
try:
    import nidaqmx  # type: ignore
    HAS_NI = True
except Exception:
    HAS_NI = False

import serial

# =============== Arduino Reader =================
class ArduinoReader(threading.Thread):
    """
    Reads 'v1,...,v16\\n' lines from Arduino, stores latest raw_avg (pre-bias),
    and keeps a rolling buffer for bias estimation.
    """
    def __init__(self, port, baud=115200, bias_window_s=2.0, sensor_rate_guess=500.0):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.stop_evt = threading.Event()
        self.ser = None
        self.latest_raw_avg = np.nan
        self._buf = bytearray()
        self.bias_vals = deque(maxlen=int(bias_window_s * sensor_rate_guess))

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.01)
            self.ser.reset_input_buffer()
            time.sleep(0.2)
            print(f"[Arduino] Opened {self.port}")
        except Exception as e:
            print(f"[Arduino] Open error: {e}")
            return
        while not self.stop_evt.is_set():
            try:
                n = self.ser.in_waiting
                if n:
                    chunk = self.ser.read(n)
                    if not chunk:
                        continue
                    self._buf.extend(chunk)
                    while True:
                        nl = self._buf.find(b'\n')
                        if nl < 0:
                            break
                        line = self._buf[:nl].decode('ascii', errors='ignore').strip()
                        del self._buf[:nl+1]
                        if not line:
                            continue
                        parts = line.split(',')
                        if len(parts) != 16:
                            continue
                        try:
                            vals = [float(p) for p in parts]
                        except Exception:
                            continue
                        avg = float(sum(vals)/16.0)
                        self.latest_raw_avg = avg
                        self.bias_vals.append(avg)
                else:
                    time.sleep(0.001)
            except Exception:
                time.sleep(0.002)

    def sensor_bias(self):
        arr = np.array([v for v in self.bias_vals if np.isfinite(v)], dtype=float)
        return 0.0 if arr.size == 0 else float(np.median(arr))

    def close(self):
        self.stop_evt.set()
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass

# =============== Bias collection helpers ===============
def collect_sensor_bias(ard: ArduinoReader, seconds: float):
    """Wait 'seconds' while ArduinoReader fills the rolling buffer, then return median."""
    t0 = time.time()
    while (time.time() - t0) < seconds:
        time.sleep(0.01)
    bias = ard.sensor_bias()
    print(f"[Sensor] Bias collection done. duration={seconds:.2f}s, bias(raw-units)={bias:.6f}")
    return bias

def collect_ft_bias(seconds: float, sample_hz: int, device: str, simulation: bool):
    """Collect FT bias as mean of voltage samples for 'seconds'."""
    dt = 1.0 / float(sample_hz)
    if simulation or (not HAS_NI):
        time.sleep(seconds)
        ft_bias_volts = np.zeros(6)
        count = int(seconds / dt)
        print(f"[FT] Bias collection done (sim). duration={seconds:.2f}s, samples≈{count}, bias(volts)={ft_bias_volts}")
        return ft_bias_volts

    import nidaqmx  # local import to avoid plugin errors if missing
    samples = []
    try:
        with nidaqmx.Task() as task:
            for ch in range(6):
                task.ai_channels.add_ai_voltage_chan(f"{device}/ai{ch}")
            t0 = time.time()
            while (time.time() - t0) < seconds:
                v = np.array(task.read())
                samples.append(v)
                time.sleep(dt)
        if not samples:
            samples.append(np.zeros(6))
        ft_bias = np.mean(samples, axis=0)
        print(f"[FT] Bias collection done. duration={seconds:.2f}s, samples={len(samples)}, bias(volts)={np.array2string(ft_bias, precision=6)}")
        return ft_bias
    except Exception as e:
        print(f"[FT] Bias error: {e}  (falling back to zeros)")
        return np.zeros(6)

# =============== Recording (5 s after Enter) ===============
def record_window(rows_sink, ard: ArduinoReader, sensor_bias, ft_bias_volts, sample_hz, seconds, device, simulation):
    dt = 1.0 / float(sample_hz)
    if simulation or (not HAS_NI):
        print("[FT] Recording (sim)…")
        t0 = time.time()
        while (time.time() - t0) < seconds:
            t = time.time() - t0
            ft_z = 2.0*math.sin(2*math.pi*0.5*t) + 0.1*np.random.normal()
            s_raw_avg = ard.latest_raw_avg
            s_raw = s_raw_avg - sensor_bias
            rows_sink.append({
                "raw_avg":   float(s_raw_avg),
                "force_c0":  float(poly_eval(COEFFS_C0,   s_raw)),
                "force_curv":float(poly_eval(COEFFS_CURV, s_raw)),
                "curvature": float(CURV),
                "ft_z":      float(ft_z),
            })
            time.sleep(dt)
        return

    import nidaqmx
    print("[FT] Recording…")
    try:
        with nidaqmx.Task() as task:
            for ch in range(6):
                task.ai_channels.add_ai_voltage_chan(f"{device}/ai{ch}")
            t0 = time.time()
            while (time.time() - t0) < seconds:
                v = np.array(task.read()) - ft_bias_volts
                ft = CALIBRATION_MATRIX @ v
                ft_z = -float(ft[2])  # positive for pressing
                s_raw_avg = ard.latest_raw_avg
                s_raw = s_raw_avg - sensor_bias
                rows_sink.append({
                    "raw_avg":   float(s_raw_avg),                         # pre-bias avg stored
                    "force_c0":  float(poly_eval(COEFFS_C0,   s_raw)),     # c=0 model
                    "force_curv":float(poly_eval(COEFFS_CURV, s_raw)),     # curvature-aware model
                    "curvature": float(CURV),
                    "ft_z":      float(ft_z),
                })
                time.sleep(dt)
    except Exception as e:
        print(f"[FT] NI-DAQ error during recording: {e}")

# =============== Save CSV =================
def save_rows(rows, curv=CURV):
    if not rows:
        print("No data captured.")
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"hold_{ts}_refNA_curv{int(curv)}.csv"
    df = pd.DataFrame(rows, columns=["raw_avg","force_c0","force_curv","curvature","ft_z"])
    df.to_csv(fname, index=False)
    print(f"Saved {len(df)} rows to {fname}")
    return fname

# =============== Main =================
if __name__ == "__main__":
    print("Starting… (collecting biases now)")
    # Start Arduino so sensor bias can accumulate immediately
    ard = ArduinoReader(COM_PORT, BAUD, bias_window_s=SENSOR_BIAS_SEC, sensor_rate_guess=500.0)
    ard.start()

    # Collect sensor bias (rolling median over SENSOR_BIAS_SEC)
    sensor_bias = collect_sensor_bias(ard, SENSOR_BIAS_SEC)

    # Collect FT bias (mean volts over FT_BIAS_SEC)
    ft_bias_volts = collect_ft_bias(FT_BIAS_SEC, SAMPLE_HZ, NI_DEVICE, SIMULATION)

    # Biases are done — print a short combined cue
    print("[System] Bias collection done. Press Enter to start 5 s recording…")

    try:
        input()  # wait for your keypress
        rows = []
        record_window(rows, ard, sensor_bias, ft_bias_volts, SAMPLE_HZ, RECORD_SEC, NI_DEVICE, SIMULATION)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        rows = []
    finally:
        ard.close()

    # (Optional) show FT bias converted to forces for reference
    try:
        ft_bias_force = CALIBRATION_MATRIX @ ft_bias_volts
        print(f"[FT] Bias (force/torque units): {np.array2string(ft_bias_force, precision=6)}  [Fx,Fy,Fz,Tx,Ty,Tz]")
    except Exception:
        pass

    save_rows(rows, CURV)
    print("Done.")
