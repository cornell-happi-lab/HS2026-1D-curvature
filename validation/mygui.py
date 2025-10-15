import sys, time, math, threading
from collections import deque
import numpy as np
import pandas as pd

from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

TOLERANCE_N = 0.1     # within ±0.1 N of reference
HOLD_SECONDS = 5.0    # must sustain for 3 s
SMA_WINDOW = 5

try:
    import nidaqmx
    HAS_NI = True
except Exception:
    HAS_NI = False

# --------------------- Config ---------------------
CALIBRATION_MATRIX = np.array([
    [0.03133, 0.10222, 0.10361, -6.63382, 0.01609, 6.57641],
    [0.21162, 8.06214, 0.15262, -3.75490, -0.15622, -3.89770],
    [7.52422, -0.09766, 7.42244, 0.10983, 7.50850, 0.21699],
    [0.47851, 48.61235, 42.25686, -21.68818, -42.88027, -24.77775],
    [-48.77394, -0.16167, 23.21350, 40.69095, 25.01598, -39.08706],
    [0.69014, 30.86257, -0.40559, 28.61573, 0.04423, 29.42892],
])

DEFAULT_DEVICE = "Dev1"
DEFAULT_SAMPLE_HZ = 1000  # internal reading cadence; GUI down-samples to display
PLOT_FPS = 40             # GUI refresh rate
BUFFER_SECONDS = 20       # rolling plot window

# --------------------- Worker Thread ---------------------
class FTWorker(QtCore.QThread):
    sample = QtCore.Signal(float, float)  # (t, z_force)
    status = QtCore.Signal(str)
    biased = QtCore.Signal(object)        # bias vector

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
            self.status.emit("Running in SIMULATION mode (no NI‑DAQ)")
            self._run_sim(dt)
            return

        # --- Hardware path ---
        try:
            with nidaqmx.Task() as task:
                for ch in range(6):
                    task.ai_channels.add_ai_voltage_chan(f"{self.device}/ai{ch}")
                # Optional: hardware timing; many USB devices okay with on-demand
                # task.timing.cfg_samp_clk_timing(self.sample_hz)

                # Collect bias
                self.status.emit("Collecting bias... hold sensor unloaded")
                bias_samples = []
                n_bias = max(100, int(0.2 * self.sample_hz))
                for _ in range(n_bias):
                    v = np.array(task.read())
                    bias_samples.append(v)
                    time.sleep(dt)
                self.bias = np.mean(bias_samples, axis=0)
                self.biased.emit(self.bias)
                self.status.emit(f"Bias set: {np.array2string(self.bias, precision=3)}")

                # Stream
                while not self._stop.is_set():
                    v = np.array(task.read())
                    if self.bias is not None:
                        v = v - self.bias
                    ft = CALIBRATION_MATRIX @ v
                    z = -float(ft[2])
                    t = time.time() - self.t0
                    self.sample.emit(t, z)
                    time.sleep(dt)
        except Exception as e:
            self.status.emit(f"NI‑DAQ error: {e}")

    def _run_sim(self, dt):
        rng = np.random.default_rng(42)
        # Simulate a slow varying force with noise
        sim_t = 0.0
        while not self._stop.is_set():
            z = 3.0 * math.sin(2 * math.pi * 0.3 * sim_t) + 0.3 * rng.normal()
            self.sample.emit(sim_t, z)
            sim_t += dt
            time.sleep(dt)

# --------------------- GUI ---------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Realtime ATI FT (Z) + Triangle Reference")
        self.resize(980, 640)

        # Central widget
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)

        # Plot
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel('left', 'Force Z', units='N')
        self.plot.setLabel('bottom', 'Time', units='s')
        self.curve_ft = self.plot.plot(pen=pg.mkPen('r', width=2))
        self.curve_ref = self.plot.plot(pen=pg.mkPen('b', style=QtCore.Qt.DashLine, width=2))
        self.legend = self.plot.addLegend()
        self.legend.addItem(self.curve_ft, "FT Z")
        self.legend.addItem(self.curve_ref, "Triangle ref")

        # Controls
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_save = QtWidgets.QPushButton("Save CSV…")
        self.btn_save.setEnabled(False)

        self.device_edit = QtWidgets.QLineEdit(DEFAULT_DEVICE)
        self.device_edit.setFixedWidth(100)
        self.sample_spin = QtWidgets.QSpinBox()
        self.sample_spin.setRange(50, 5000)
        self.sample_spin.setValue(DEFAULT_SAMPLE_HZ)
        self.sample_spin.setSuffix(" Hz")

        self.freq_dbl = QtWidgets.QDoubleSpinBox()
        self.freq_dbl.setRange(0.01, 10.0)
        self.freq_dbl.setValue(0.5)
        self.freq_dbl.setSingleStep(0.05)
        self.freq_dbl.setSuffix(" Hz")

        self.amp_dbl = QtWidgets.QDoubleSpinBox()
        self.amp_dbl.setRange(0.0, 1000.0)
        self.amp_dbl.setValue(3.0)
        self.amp_dbl.setSingleStep(0.5)
        self.amp_dbl.setSuffix(" N")

        self.off_dbl = QtWidgets.QDoubleSpinBox()
        self.off_dbl.setRange(-1000.0, 1000.0)
        self.off_dbl.setValue(0.0)
        self.off_dbl.setSingleStep(0.5)
        self.off_dbl.setSuffix(" N")

        self.sim_check = QtWidgets.QCheckBox("Simulation")
        self.sim_check.setChecked(not HAS_NI)
        self.sim_check.setToolTip("Run without NI‑DAQ (synthetic signal)")

        self.status_lab = QtWidgets.QLabel("Idle")

        # Layouts
        form = QtWidgets.QFormLayout()
        form.addRow("Device:", self.device_edit)
        form.addRow("Sample rate:", self.sample_spin)
        form.addRow("Ref freq:", self.freq_dbl)
        form.addRow("Ref amplitude:", self.amp_dbl)
        form.addRow("Ref offset:", self.off_dbl)
        form.addRow(self.sim_check)

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self.btn_start)
        btns.addWidget(self.btn_stop)
        btns.addWidget(self.btn_save)

        right = QtWidgets.QVBoxLayout()
        right.addLayout(form)
        right.addStretch(1)
        right.addLayout(btns)
        right.addWidget(self.status_lab)

        lay = QtWidgets.QHBoxLayout(cw)
        lay.addWidget(self.plot, stretch=1)
        lay.addLayout(right)

        # Data buffers
        self.max_points = int(BUFFER_SECONDS * DEFAULT_SAMPLE_HZ)
        self.ts = deque(maxlen=self.max_points)
        self.zs = deque(maxlen=self.max_points)
        self.all_rows = []  # for saving

        # Stage logic: levels are fractions of amplitude (0..1)
        self.stage_levels = [0.0, 0.5, 1.0, 0.0]  # baseline → medium → strong → baseline
        self.stage_index = 0
        self._match_start = None  # when we first entered tolerance window

        # Worker & timer
        self.worker = None
        self.gui_timer = QtCore.QTimer()
        self.gui_timer.setInterval(int(1000 / PLOT_FPS))
        self.gui_timer.timeout.connect(self._on_gui_timer)

        # Signals
        self.btn_start.clicked.connect(self.start_acq)
        self.btn_stop.clicked.connect(self.stop_acq)
        self.btn_save.clicked.connect(self.save_csv)

    # ---------- Acquisition Control ----------
    def start_acq(self):
        if self.worker is not None:
            return
        self.ts.clear(); self.zs.clear(); self.all_rows.clear()
        device = self.device_edit.text().strip()
        sr = self.sample_spin.value()
        simulate = self.sim_check.isChecked()

        self.worker = FTWorker(device=device, sample_hz=sr, simulate=simulate)
        self.worker.sample.connect(self._on_sample)
        self.worker.status.connect(self._set_status)
        self.worker.biased.connect(lambda b: None)
        self.worker.start()

        self.stage_index = 0
        self._match_start = None

        self.t_plot0 = time.time()
        self.gui_timer.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_save.setEnabled(False)
        self._set_status("Acquiring…")

    def stop_acq(self):
        if self.worker is None:
            return
        self.worker.stop()
        self.worker.wait(2000)
        self.worker = None
        self.gui_timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_save.setEnabled(True)
        self._set_status("Stopped")

    # ---------- Data / Plot ----------
    @QtCore.Slot(float, float)
    def _on_sample(self, t, z):
        self.ts.append(t)
        self.zs.append(z)
        self.all_rows.append((t, z))

    # def _triangle(self, t):
    #     """Triangle wave in [-1,1] at time t (seconds) with frequency f."""
    #     f = max(1e-6, self.freq_dbl.value())
    #     phase = (t * f) % 1.0
    #     tri = 2.0 * abs(2.0 * phase - 1.0) - 1.0  # [-1,1]
    #     return tri
    
    def _step_pattern(self, t):
        cycle = 30.0
        phase = t % cycle
        if phase < 5:
            level = 0.1
        elif phase < 10:
            level = 0.3
        elif phase < 15:
            level = 0.5
        elif phase < 20:
            level = 0.7
        elif phase < 25:
            level = 0.9
        else:
            level = 0.0
        return level

    def _on_gui_timer(self):
        if not self.ts:
            return
        t = np.fromiter(self.ts, dtype=float)
        z = np.fromiter(self.zs, dtype=float)

        WIN = SMA_WINDOW
        if z.size >= WIN:
            k = (WIN - 1) // 2
            kernel = np.ones(WIN, dtype=float) / WIN
            # pad with edge values instead of zeros to avoid end sag
            z_pad = np.pad(z, (k, k), mode='edge')   # or mode='reflect'
            z_smooth = np.convolve(z_pad, kernel, mode='valid')  # same length as z
        else:
            z_smooth = z

        # # Triangle reference evaluated at same timestamps for visual sync
        # tri = np.array([self._triangle(tt) for tt in t])
        # ref = self.off_dbl.value() + self.amp_dbl.value() * tri
        # steps = np.array([self._step_pattern(tt) for tt in t])
        # ref = self.off_dbl.value() + self.amp_dbl.value() * steps
        # Current stage's flat reference level
        level = self.stage_levels[self.stage_index]          # 0.0, 0.5, 1.0, ...
        ref = self.off_dbl.value() + self.amp_dbl.value() * level
        ref_arr = np.full_like(t, ref, dtype=float)

        # ---- Advance stage if FT stays within tolerance for HOLD_SECONDS ----
        if t.size > 0:
            now = t[-1]
            z_now = z_smooth[-1]
            if abs(z_now - ref) <= TOLERANCE_N:
                if self._match_start is None:
                    self._match_start = now
                elif (now - self._match_start) >= HOLD_SECONDS:
                    # Advance to next stage (wrap)
                    self.stage_index = (self.stage_index + 1) % len(self.stage_levels)
                    self._match_start = None
                    next_ref = self.off_dbl.value() + self.amp_dbl.value() * self.stage_levels[self.stage_index]
                    self._set_status(f"Stage → {self.stage_index} (ref={next_ref:.2f} N)")
            else:
                self._match_start = None


        # Limit x-range to rolling window
        t0 = t[-1] - BUFFER_SECONDS
        self.plot.setXRange(max(0.0, t0), t[-1])

        # Plot
        self.curve_ft.setData(t, z_smooth)
        self.curve_ref.setData(t, ref_arr)
        # self.curve_ft.setData(t, z)
        # self.curve_ref.setData(t, ref)

    # ---------- Utils ----------
    def _set_status(self, s):
        self.status_lab.setText(s)

    def save_csv(self):
        if not self.all_rows:
            QtWidgets.QMessageBox.information(self, "Save CSV", "No data to save.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save CSV", "ft_stream.csv", "CSV Files (*.csv)")
        if not path:
            return
        df = pd.DataFrame(self.all_rows, columns=["time_s", "ft_z"])
        try:
            df.to_csv(path, index=False)
            self._set_status(f"Saved: {path}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Save CSV", f"Failed to save: {e}")

# --------------------- main ---------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # Nice default look for pyqtgraph
    pg.setConfigOptions(antialias=True)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
