import sys, time, math, threading
from collections import deque
import numpy as np
import pandas as pd

from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg
import serial

# ========= User-config =========
COM_PORT = "COM5"              # Arduino COM port
DEFAULT_DEVICE = "Dev1"        # NI-DAQ device name
DEFAULT_SAMPLE_HZ = 1000
PLOT_FPS = 40
BUFFER_SECONDS = 20

# Reference/logic
TOLERANCE_N = 0.2              # within ±0.2 N of reference
HOLD_SECONDS = 5.0             # must sustain for 5 s
EMA_ALPHA = 0.3

# ========= ATI calibration =========
CALIBRATION_MATRIX = np.array([
    [0.03133, 0.10222, 0.10361, -6.63382, 0.01609, 6.57641],
    [0.21162, 8.06214, 0.15262, -3.75490, -0.15622, -3.89770],
    [7.52422, -0.09766, 7.42244, 0.10983, 7.50850, 0.21699],
    [0.47851, 48.61235, 42.25686, -21.68818, -42.88027, -24.77775],
    [-48.77394, -0.16167, 23.21350, 40.69095, 25.01598, -39.08706],
    [0.69014, 30.86257, -0.40559, 28.61573, 0.04423, 29.42892],
])

# ========= COEFFS CAL =========
def coeffs_from_curvature(c: float):
    """
    Given curvature c, return SENSOR_POLY_COEFFS for np.polyval
    Model (only terms including Sensor, F(0,c)=0 by construction):
        F = + 0.009625 * Sensor
            - 0.000372 * Curvature * Sensor
            + 0.000050 * Curvature^2 * Sensor
            - 0.000014 * Sensor^2
            + 0.000000 * Sensor^3
            + 0.000000
    """
    a3 = 0.0
    a2 = -0.000014
    a1 = 0.009625 + (-0.000372 * c) + (0.000050 * (c**2))
    a0 = 0.0

    # If you conceptually sum four blocks, scale here.
    # Your last version used *6; keep that behavior:
    scale = 4.0
    return [scale * a3, scale * a2, scale * a1, scale * a0]

CURV = 42.28
SENSOR_POLY_COEFFS    = coeffs_from_curvature(CURV)
SENSOR_POLY_COEFFS_C0 = coeffs_from_curvature(0.0)

# ========= NI availability =========
try:
    import nidaqmx
    HAS_NI = True
except Exception:
    HAS_NI = False


# ================= Worker Thread (NI-DAQ) =================
class FTWorker(QtCore.QThread):
    sample = QtCore.Signal(float, float)  # (t, z_force)
    status = QtCore.Signal(str)
    biased = QtCore.Signal(object)        # emit bias vector when done

    def __init__(self, device=DEFAULT_DEVICE, sample_hz=DEFAULT_SAMPLE_HZ, simulate=False, parent=None):
        super().__init__(parent)
        self.device = device
        self.sample_hz = sample_hz
        self.simulate = simulate or (not HAS_NI)
        self._stop = threading.Event()
        self.bias = None
        self.t0 = None

    def stop(self):
        self._stop.set()

    def run(self):
        self._stop.clear()
        self.t0 = time.time()
        dt = 1.0 / float(self.sample_hz)

        if self.simulate:
            self.status.emit("Running in SIMULATION mode (no NI-DAQ)")
            # brief bias
            time.sleep(0.3)
            self.bias = np.zeros(6)
            self.biased.emit(self.bias)
            self._run_sim(dt)
            return

        try:
            with nidaqmx.Task() as task:
                for ch in range(6):
                    task.ai_channels.add_ai_voltage_chan(f"{self.device}/ai{ch}")

                # Collect bias
                self.status.emit("Collecting FT bias... hold sensor unloaded")
                bias_samples = []
                n_bias = max(200, int(0.3 * self.sample_hz))
                for _ in range(n_bias):
                    if self._stop.is_set():
                        return
                    v = np.array(task.read())
                    bias_samples.append(v)
                    time.sleep(dt)
                self.bias = np.mean(bias_samples, axis=0)
                self.biased.emit(self.bias)
                self.status.emit("Streaming…")

                # Stream
                while not self._stop.is_set():
                    v = np.array(task.read())
                    if self.bias is not None:
                        v = v - self.bias
                    ft = CALIBRATION_MATRIX @ v
                    z = -float(ft[2])  # invert so presses are positive
                    t = time.time() - self.t0
                    self.sample.emit(t, z)
                    time.sleep(dt)
        except Exception as e:
            self.status.emit(f"NI-DAQ error: {e}")

    def _run_sim(self, dt):
        rng = np.random.default_rng(42)
        sim_t = 0.0
        while not self._stop.is_set():
            z = 2.0 * math.sin(2 * math.pi * 0.3 * sim_t) + 0.2 * rng.normal()
            self.sample.emit(sim_t, z)
            sim_t += dt
            time.sleep(dt)


# ============================ GUI ============================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Two-Plot Realtime GUI — Sensor (top) / FT+Ref (bottom)")
        self.resize(1200, 800)

        # Central widget & main layout
        cw = QtWidgets.QWidget(); self.setCentralWidget(cw)
        main_v = QtWidgets.QVBoxLayout(cw)

        # Shared state for Arduino reader
        self.sensor_lock = threading.Lock()
        self._arduino = None
        self._arduino_thread = None
        self._arduino_stop = threading.Event()

        # Epoch for plotting
        self.t0_ui = None

        # ---- Top plot: Sensor (calibrated, own timestamps) ----
        self.plot_sensor = pg.PlotWidget()
        self.plot_sensor.showGrid(x=True, y=True, alpha=0.3)
        self.plot_sensor.setLabel('left', 'Sensor Force', units='N')
        self.plot_sensor.getAxis('left').enableAutoSIPrefix(False)
        self.plot_sensor.setLabel('bottom', 'Time', units='s')
        self.curve_sensor    = self.plot_sensor.plot(pen=pg.mkPen('g', width=2))
        self.curve_sensor_c0 = self.plot_sensor.plot(pen=pg.mkPen('m', style=QtCore.Qt.DashLine, width=2))

        self.plot_sensor.addLegend()
        self.plot_sensor.getPlotItem().legend.addItem(self.curve_sensor,    f"Sensor (CURV={int(CURV)})")
        self.plot_sensor.getPlotItem().legend.addItem(self.curve_sensor_c0, "Sensor (C=0)")

        # ---- Bottom plot: FT + Reference (FT timestamps) ----
        self.plot_ft = pg.PlotWidget()
        self.plot_ft.showGrid(x=True, y=True, alpha=0.3)
        self.plot_ft.setLabel('left', 'FT Z / Ref', units='N')
        self.plot_ft.getAxis('left').enableAutoSIPrefix(False)
        self.plot_ft.setLabel('bottom', 'Time', units='s')
        self.curve_ft  = self.plot_ft.plot(pen=pg.mkPen('r', width=2))
        self.curve_ref = self.plot_ft.plot(pen=pg.mkPen('b', style=QtCore.Qt.DashLine, width=2))
        self.plot_ft.addLegend()
        self.plot_ft.getPlotItem().legend.addItem(self.curve_ft, "FT Z")
        self.plot_ft.getPlotItem().legend.addItem(self.curve_ref, "Reference")
        # Lock Y to 0–5 N
        # self.plot_ft.getPlotItem().setYRange(0, 5)
        self.plot_ft.getPlotItem().enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

        main_v.addWidget(self.plot_sensor, stretch=1)
        main_v.addWidget(self.plot_ft, stretch=1)

        # ---- Controls ----
        ctrl = QtWidgets.QWidget(); main_v.addWidget(ctrl)
        form = QtWidgets.QFormLayout(ctrl)

        self.device_edit = QtWidgets.QLineEdit(DEFAULT_DEVICE); self.device_edit.setFixedWidth(140)
        self.sample_spin = QtWidgets.QSpinBox(); self.sample_spin.setRange(50, 5000)
        self.sample_spin.setValue(DEFAULT_SAMPLE_HZ); self.sample_spin.setSuffix(" Hz")

        self.amp_dbl = QtWidgets.QDoubleSpinBox(); self.amp_dbl.setRange(0.0, 1000.0)
        self.amp_dbl.setValue(8.0); self.amp_dbl.setSingleStep(0.5); self.amp_dbl.setSuffix(" N")
        self.off_dbl = QtWidgets.QDoubleSpinBox(); self.off_dbl.setRange(-1000.0, 1000.0)
        self.off_dbl.setValue(0.0); self.off_dbl.setSingleStep(0.5); self.off_dbl.setSuffix(" N")

        self.sim_check = QtWidgets.QCheckBox("Simulation"); self.sim_check.setChecked(not HAS_NI)
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop  = QtWidgets.QPushButton("Stop"); self.btn_stop.setEnabled(False)
        self.btn_save  = QtWidgets.QPushButton("Save CSV…"); self.btn_save.setEnabled(False)

        # Sensor bias controls
        self.sensor_bias_btn = QtWidgets.QPushButton("Zero sensor (bias)")
        self.sensor_bias_lbl = QtWidgets.QLabel("bias: 0.0000")

        self.status_lab = QtWidgets.QLabel("Idle")

        form.addRow("Device:", self.device_edit)
        form.addRow("Sample rate:", self.sample_spin)
        form.addRow("Ref amplitude:", self.amp_dbl)
        form.addRow("Ref offset:",   self.off_dbl)
        form.addRow(self.sim_check)
        form.addRow(self.sensor_bias_btn, self.sensor_bias_lbl)

        h = QtWidgets.QHBoxLayout()
        h.addWidget(self.btn_start); h.addWidget(self.btn_stop); h.addWidget(self.btn_save)
        form.addRow(h)
        form.addRow(self.status_lab)

        # Data buffers (separate time bases)
        self.max_points = int(BUFFER_SECONDS * DEFAULT_SAMPLE_HZ)
        # FT
        self.ft_ts = deque(maxlen=self.max_points)
        self.ft_zs = deque(maxlen=self.max_points)
        # Sensor (own timestamps)
        self.sensor_ts   = deque(maxlen=self.max_points)
        self.sensor_vals = deque(maxlen=self.max_points)  # raw avg (before bias)
        # CSV buffers (manual save)
        self.rows_ft = []       # (t, ft_z, ref)
        self.rows_sensor = []   # (t, sensor_force)

        # Stage logic: fractions of amplitude (0..1)
        self.stage_levels = [0.0, 0.25, 0.50, 0.75, 1.0, 0.0]
        self.stage_index = 0
        self._match_start = None

        # Hold capture (auto CSV)
        self._hold_recording = False
        self._hold_rows = []   # each row: raw_avg, force_c0, force_curv, curvature, ft_z

        # Sensor bias
        self.sensor_bias = 0.0

        # Worker & timer
        self.worker = None
        self.gui_timer = QtCore.QTimer(); self.gui_timer.setInterval(int(1000 / PLOT_FPS))
        self.gui_timer.timeout.connect(self._on_gui_timer)

        # Signals
        self.btn_start.clicked.connect(self.start_acq)
        self.btn_stop.clicked.connect(self.stop_acq)
        self.btn_save.clicked.connect(self.save_csv)
        self.sensor_bias_btn.clicked.connect(self._sensor_set_bias)

    # ---------- Acquisition Control ----------
    def start_acq(self):
        if self.worker is not None:
            return

        # Clear buffers
        self.ft_ts.clear(); self.ft_zs.clear()
        self.sensor_ts.clear(); self.sensor_vals.clear()
        self.rows_ft.clear(); self.rows_sensor.clear()
        self._hold_rows.clear(); self._hold_recording = False
        self._match_start = None

        device = self.device_edit.text().strip()
        sr = self.sample_spin.value()
        simulate = self.sim_check.isChecked()

        # Common epoch for plots
        self.t0_ui = time.time()

        # Start Arduino FIRST so we have sensor data during FT biasing
        self._start_arduino(port=COM_PORT, baud=115200)

        # Start FT worker
        self.worker = FTWorker(device=device, sample_hz=sr, simulate=simulate, parent=self)
        self.worker.sample.connect(self._on_ft_sample)
        self.worker.status.connect(self._set_status)
        self.worker.biased.connect(self._on_ft_biased)
        self.worker.start()

        # Reset stage state
        self.stage_index = 0

        self.gui_timer.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_save.setEnabled(False)
        self._set_status("Acquiring…")

    def stop_acq(self):
        if self.worker is None:
            return
        self.worker.stop()
        self._stop_arduino()
        self.worker.wait(2000)
        self.worker = None
        self.gui_timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_save.setEnabled(True)
        self._set_status("Stopped")

    # ---------- Slots ----------
    @QtCore.Slot(object)
    def _on_ft_biased(self, _bias_vec):
        """Initial sensor bias from recent sensor values (robust ~2 s median)."""
        if len(self.sensor_ts) == 0:
            self.sensor_bias = 0.0
        else:
            now = self.sensor_ts[-1]
            t_start = now - 2.0
            idx = np.where(np.array(self.sensor_ts) >= t_start)[0]
            if idx.size:
                vals = np.array([self.sensor_vals[i] for i in idx], dtype=float)
                vals = vals[np.isfinite(vals)]
                self.sensor_bias = float(np.nanmedian(vals)) if vals.size else 0.0
            else:
                self.sensor_bias = 0.0
        self.sensor_bias_lbl.setText(f"bias: {self.sensor_bias:.4f}")
        self._set_status(f"Sensor bias set: {self.sensor_bias:.4f}")

    @QtCore.Slot(float, float)
    def _on_ft_sample(self, t, ft_z):
        self.ft_ts.append(t)
        self.ft_zs.append(ft_z)

    # ---------- Helpers ----------
    def _ema(self, arr, alpha=EMA_ALPHA):
        if arr.size == 0:
            return arr
        out = np.empty_like(arr, dtype=float)
        out[0] = arr[0]
        for i in range(1, arr.size):
            out[i] = alpha * arr[i] + (1.0 - alpha) * out[i-1]
        return out

    def _step_level(self):
        return self.stage_levels[self.stage_index]

    def _append_hold_row(self, sensor_avg, sensor_force_c0, sensor_force_curv, ft_z):
        """Append one row (at current sample) to the hold buffer."""
        row = {
            "raw_avg":   float(sensor_avg),
            "force_c0":  float(sensor_force_c0),
            "force_curv":float(sensor_force_curv),
            "curvature": float(CURV),
            "ft_z":      float(ft_z),
        }
        self._hold_rows.append(row)

    def _save_hold_csv(self, ref_value):
        """Write the captured 5 s window to CSV with an auto filename."""
        if not self._hold_rows:
            return
        df = pd.DataFrame(self._hold_rows)
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"hold_{ts}_ref{ref_value:.2f}_curv{int(CURV)}.csv"
        try:
            df.to_csv(fname, index=False)
            self._set_status(f"Saved hold window to {fname}")
        except Exception as e:
            self._set_status(f"Save error: {e}")

    # ---------- GUI update ----------
    def _on_gui_timer(self):
        # --- Copy FT arrays ---
        t_ft = np.fromiter(self.ft_ts, dtype=float)
        ft = np.fromiter(self.ft_zs, dtype=float)

        # --- Copy Sensor arrays (own timestamps) ---
        s_t = np.fromiter(self.sensor_ts, dtype=float)
        s_raw_before_bias = np.fromiter(self.sensor_vals, dtype=float)

        # --- Sensor: bias & calibration, then EMA ---
        s_raw = s_raw_before_bias - self.sensor_bias
        s_force     = np.polyval(SENSOR_POLY_COEFFS,    s_raw)
        s_force_c0  = np.polyval(SENSOR_POLY_COEFFS_C0, s_raw)
        s_force_sm    = self._ema(s_force, alpha=EMA_ALPHA)
        s_force_c0_sm = self._ema(s_force_c0, alpha=EMA_ALPHA)

        # --- FT: EMA ---
        ft_sm = self._ema(ft, alpha=EMA_ALPHA)

        # --- Reference using FT timebase (flat per stage) ---
        if t_ft.size:
            level = self._step_level()
            ref = self.off_dbl.value() + self.amp_dbl.value() * level
            ref_arr = np.full_like(t_ft, ref, dtype=float)
        else:
            ref = 0.0
            ref_arr = np.array([], dtype=float)

        # --- Stage advance (continuous hold) on FT stream + hold recording ---
        if t_ft.size > 0 and ft_sm.size > 0:
            now = t_ft[-1]
            in_tol = abs(ft_sm[-1] - ref) <= TOLERANCE_N

            if in_tol:
                # Start timing & recording if needed
                if self._match_start is None:
                    self._match_start = now
                    self._hold_recording = True
                    self._hold_rows.clear()

                # Append one row for the current sample (use latest values)
                if s_t.size:
                    sensor_avg_latest  = s_raw[-1] + self.sensor_bias  # store *raw avg* (pre-bias) or post-bias?
                    # You asked for "raw_average" — use bias-subtracted or original?
                    # You said "raw_average" (sensor's raw reading). We'll store the *pre-bias* average:
                    sensor_avg_latest = s_raw_before_bias[-1]

                    sensor_c0_latest  = s_force_c0[-1]
                    sensor_curv_latest= s_force[-1]
                    ft_latest         = ft_sm[-1]
                    self._append_hold_row(sensor_avg_latest, sensor_c0_latest, sensor_curv_latest, ft_latest)

                # Save and advance if we reached hold duration
                if (now - self._match_start) >= HOLD_SECONDS:
                    self._save_hold_csv(ref_value=ref)
                    self._hold_recording = False
                    self._hold_rows.clear()

                    # Advance stage (wrap)
                    self.stage_index = (self.stage_index + 1) % len(self.stage_levels)
                    self._match_start = None
                    next_ref = self.off_dbl.value() + self.amp_dbl.value() * self._step_level()
                    self._set_status(f"Stage → {self.stage_index} (ref={next_ref:.2f} N)")
            else:
                # Left tolerance early: reset recording and timer
                self._match_start = None
                if self._hold_recording:
                    self._hold_recording = False
                    self._hold_rows.clear()

        # --- X windowing per-plot ---
        if s_t.size:
            t0s = s_t[-1] - BUFFER_SECONDS
            self.plot_sensor.setXRange(max(0.0, t0s), s_t[-1])
        if t_ft.size:
            t0f = t_ft[-1] - BUFFER_SECONDS
            self.plot_ft.setXRange(max(0.0, t0f), t_ft[-1])

        # --- Sensor Y auto-range; FT Y stays locked 0–5 ---
        if s_force_sm.size or s_force_c0_sm.size:
            vals = []
            if s_force_sm.size:    vals += [np.nanmin(s_force_sm),    np.nanmax(s_force_sm)]
            if s_force_c0_sm.size: vals += [np.nanmin(s_force_c0_sm), np.nanmax(s_force_c0_sm)]
            ys_min = float(np.nanmin(vals)); ys_max = float(np.nanmax(vals))
            span = max(1e-6, ys_max - ys_min); pad = 0.1 * span
            self.plot_sensor.setYRange(ys_min - pad, ys_max + pad)

        # --- Plot ---
        self.curve_sensor.setData(s_t,    s_force_sm)
        self.curve_sensor_c0.setData(s_t, s_force_c0_sm)
        self.curve_ft.setData(t_ft,  ft_sm)
        self.curve_ref.setData(t_ft, ref_arr)

        # --- CSV buffers (manual save last-only for each stream) ---
        if t_ft.size and ft_sm.size:
            self.rows_ft.append((t_ft[-1], float(ft_sm[-1]), float(ref)))
        if s_t.size and s_force_sm.size:
            self.rows_sensor.append((s_t[-1], float(s_force_sm[-1])))

    # ---------- Utils ----------
    def _set_status(self, s: str):
        self.status_lab.setText(s)

    def _sensor_set_bias(self):
        """
        Bias so that the current sensor reading (avg) becomes zero after subtraction.
        """
        with self.sensor_lock:
            if not self.sensor_ts or not self.sensor_vals:
                self._set_status("No sensor data yet to bias.")
                return
            latest_avg = float(self.sensor_vals[-1])
        self.sensor_bias = latest_avg
        self.sensor_bias_lbl.setText(f"bias: {self.sensor_bias:.4f}")
        self._set_status(f"Sensor bias updated: subtract {self.sensor_bias:.4f}")

    # ---------- Arduino handling ----------
    def _start_arduino(self, port=COM_PORT, baud=115200):
        try:
            self._arduino_stop.clear()
            self._arduino = serial.Serial(port, baud, timeout=0.01)
            self._arduino.reset_input_buffer()
            time.sleep(0.3)  # short settle
            self._arduino_thread = threading.Thread(target=self._arduino_loop, daemon=True)
            self._arduino_thread.start()
            self._set_status(f"Arduino started on {port}")
        except Exception as e:
            self._set_status(f"Arduino error: {e}")
            self._arduino = None
            self._arduino_thread = None

    def _stop_arduino(self):
        try:
            self._arduino_stop.set()
            if self._arduino_thread:
                self._arduino_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self._arduino and self._arduino.is_open:
                self._arduino.close()
        except Exception:
            pass
        self._arduino = None
        self._arduino_thread = None

    def _arduino_loop(self):
        """
        Reads 'v1,...,v16\\n' lines. Appends timestamped avg to sensor_ts/vals.
        """
        buf = bytearray()
        while not self._arduino_stop.is_set():
            try:
                n = self._arduino.in_waiting
                if n:
                    chunk = self._arduino.read(n)
                    if not chunk:
                        continue
                    buf.extend(chunk)
                    while True:
                        nl = buf.find(b'\n')
                        if nl < 0:
                            break
                        line = buf[:nl].decode('ascii', errors='ignore').strip()
                        del buf[:nl+1]
                        if not line:
                            continue
                        parts = line.split(',')
                        if len(parts) != 16:
                            continue
                        try:
                            vals = [float(p) for p in parts]
                        except Exception:
                            continue
                        avg = float(sum(vals) / 16.0)
                        t_rel = (time.time() - (self.t0_ui or time.time()))
                        with self.sensor_lock:
                            self.sensor_ts.append(t_rel)
                            self.sensor_vals.append(avg)
                else:
                    time.sleep(0.001)
            except Exception:
                time.sleep(0.001)
                continue

    # ---------- Save CSV (manual) ----------
    def save_csv(self):
        if not self.rows_ft and not self.rows_sensor:
            QtWidgets.QMessageBox.information(self, "Save CSV", "No data to save.")
            return
        base, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save CSV", "session", "CSV Files (*.csv)")
        if not base:
            return
        try:
            if self.rows_ft:
                df_ft = pd.DataFrame(self.rows_ft, columns=["time_s", "ft_z", "ref"])
                df_ft.to_csv(base.replace('.csv','') + "_ft.csv", index=False)
            if self.rows_sensor:
                df_s = pd.DataFrame(self.rows_sensor, columns=["time_s", "sensor_force"])
                df_s.to_csv(base.replace('.csv','') + "_sensor.csv", index=False)
            self._set_status(f"Saved: {base}_ft.csv and {base}_sensor.csv")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Save CSV", f"Failed to save: {e}")


# ============================ main ============================
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
