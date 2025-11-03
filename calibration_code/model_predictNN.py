# constrained_poly_calibration_with_3d_plot.py
#
# Calibrate a polynomial model with constraints:
#  - F(0, C) = 0  for ANY curvature C  (no intercept, no pure-curvature terms)
#  - No negative predictions (clamp at 0 for evaluation/visualization)
#
# Outputs:
#  - Printed constrained model equation (only terms that include Sensor)
#  - R² (unclipped) and R² (clipped at 0)
#  - Helper: coeffs_from_curvature(c) -> [a3, a2, a1, a0] for np.polyval
#  - 3D plot (scatter of data + orange surface of constrained model)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# === Step 1: Aggregate all data ===
def build_df(avg_ft, avg_sensor, curvature):
    return pd.DataFrame({
        'Force': avg_ft.values,
        'Sensor': avg_sensor.values,
        'Curvature': curvature
    })

# ---- Load and aggregate (edit file paths if needed) ----
def load_block_df(csv_path):
    df = pd.read_csv(csv_path)
    df['FT_Z'] = -df['FT_Z']  # make presses positive
    # Example block channels; adjust if yours differ
    df['block'] = df['S5'] + df['S6'] + df['S9'] + df['S10']
    grouped = df.groupby('Step')
    avg_ft = grouped['FT_Z'].mean()
    avg_sensor = grouped['block'].mean()
    return avg_ft, avg_sensor

cy_files = [
    ("calibration_data/curvature_aware_cal/sensor5_cy_0.csv",  0),
    ("calibration_data/curvature_aware_cal/sensor5_cy_20.csv", 20),
    ("calibration_data/curvature_aware_cal/sensor5_cy_40.csv", 40),
    ("calibration_data/curvature_aware_cal/sensor5_cy_60.csv", 60),
    ("calibration_data/curvature_aware_cal/sensor5_cy_80.csv", 80),
    # ("calibration_data/sensor5_cy_100.csv", 100),  # add back if available
]

dfs = []
for path, curv in cy_files:
    avg_ft, avg_sensor = load_block_df(path)
    dfs.append(build_df(avg_ft, avg_sensor, curvature=curv))

df_all = pd.concat(dfs, ignore_index=True)

# === Step 2: Create the constrained polynomial model ===
X = df_all[['Sensor', 'Curvature']].to_numpy()
y = df_all['Force'].to_numpy()

# Degree-3 polynomial features, no bias (we control intercept manually)
poly = PolynomialFeatures(degree=3, include_bias=False)
Phi = poly.fit_transform(X)
feat_names = poly.get_feature_names_out(['Sensor', 'Curvature'])

# Constraint: keep ONLY features that CONTAIN 'Sensor'
# (so plugging Sensor=0 forces prediction to 0 for any curvature)
mask = np.array(['Sensor' in name for name in feat_names])
Phi_S = Phi[:, mask]
feat_names_S = feat_names[mask]

# Fit without intercept so F(0, C) = 0 exactly
lin = LinearRegression(fit_intercept=False)
lin.fit(Phi_S, y)

# === Helper: safe prediction with non-negativity ===
def predict_force(sensor, curvature):
    x = np.array([[sensor, curvature]], dtype=float)
    phi = poly.transform(x)[:, mask]
    pred = float(lin.predict(phi)[0])
    return max(0.0, pred)  # clamp negatives to zero

# === Pretty-print the constrained equation (no intercept term) ===
terms = [f"{coef:.6f}*{name}" for coef, name in zip(lin.coef_, feat_names_S)]
equation = "Predicted Force = " + " + ".join(terms)  # F(0,C)=0 by construction
print(equation)

# === Step 3: Evaluate ===
y_pred_unclipped = lin.predict(Phi_S)
y_pred_clipped = np.maximum(0.0, y_pred_unclipped)

print(f"R² (unclipped): {r2_score(y, y_pred_unclipped):.4f}")
print(f"R² (clipped @ 0): {r2_score(y, y_pred_clipped):.4f}")

# Quick sanity checks
ex = predict_force(sensor=500, curvature=60)
print(f"Predicted force for sensor=500, curvature=60: {ex:.3f} N")
baseline0 = predict_force(sensor=0, curvature=0)
baselineC = predict_force(sensor=0, curvature=60)
print(f"Predicted force for sensor=0, curvature=0:  {baseline0:.3f} N  (should be 0)")
print(f"Predicted force for sensor=0, curvature=60: {baselineC:.3f} N  (should be 0)")

# === Step 4: Build SENSOR_POLY_COEFFS for a given curvature ===
# Returns [a3, a2, a1, a0] so you can use: np.polyval(SENSOR_POLY_COEFFS, sensor)
def coeffs_from_curvature(c: float):
    """
    For fixed curvature c, convert the trained model into a cubic in Sensor:
        F(S, c) = a3*S^3 + a2*S^2 + a1*S + a0
    Because we removed intercept and pure-curvature terms, a0 == 0.
    """
    coef_map = {name: w for name, w in zip(feat_names_S, lin.coef_)}

    def w(name):
        return float(coef_map.get(name, 0.0))

    # Allowed terms (degree<=3) that contain Sensor:
    #  S, S^2, S^3, S*C, S*C^2, S^2*C
    a3 = w('Sensor^3')  # may be 0 if not learned
    a2 = w('Sensor^2') + w('Sensor^2 Curvature') * c
    a1 = (
        w('Sensor') +
        w('Sensor Curvature')   * c +
        w('Sensor Curvature^2') * (c**2)
    )
    a0 = 0.0  # by construction

    return [a3, a2, a1, a0]

# Example usage for GUI:
SENSOR_POLY_COEFFS = coeffs_from_curvature(60.0)
print("SENSOR_POLY_COEFFS (curv=60):", SENSOR_POLY_COEFFS)
# Example of evaluating as a 1D poly in Sensor:
ex_poly = np.polyval(SENSOR_POLY_COEFFS, 500.0)
print(f"Polyval at sensor=500 (curv=60): {max(0.0, ex_poly):.3f} N")

# === Step 5: 3D Plot (same style as your original) ===
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter actual data
ax.scatter(df_all['Sensor'], df_all['Curvature'], y, label='True', c='blue', alpha=0.5)

# Surface grid (match your display ranges)
sensor_range = np.linspace(0, 1200, 40)   # X-axis (Sensor Readings)
curv_range   = np.linspace(0,   80, 40)   # Y-axis (Curvature)
SENSOR, CURV = np.meshgrid(sensor_range, curv_range)

# Predict on the grid using the constrained model
X_grid   = np.c_[SENSOR.ravel(), CURV.ravel()]
Phi_grid = poly.transform(X_grid)[:, mask]
Z        = lin.predict(Phi_grid).reshape(SENSOR.shape)
Z        = np.maximum(0.0, Z)  # enforce non-negativity for visualization

# Draw the surface
ax.plot_surface(SENSOR, CURV, Z, alpha=0.6, color='orange', edgecolor='none')

# Labels and limits to match your original look
ax.set_xlabel('Sensor Readings')
ax.set_ylabel('Curvature (1/m)')
ax.set_zlabel('FT Z Force (N)')

ax.set_xlim(0, 1200)   # Sensor range
ax.set_ylim(0,   80)   # Curvature range
ax.set_zlim(0,   30)   # Force range

ax.set_title('Constrained Model: Force ~ poly(Sensor, Curvature), F(0,C)=0, no negatives')
plt.tight_layout()
plt.show()
