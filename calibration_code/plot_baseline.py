import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Curvatures
curv_cylinder = np.array([0, 20, 40, 60, 80])
curv_sphere = np.array([0, 21, 42, 63, 84])

# Sensor readings (mean only)
mean_cylinder = np.array([732.613125, 742.3975, 757.379375, 763.65375, 776.80875])
mean_sphere = np.array([706.62, 718.18, 730.13, 740.50, 757.67])

# === Linear regression models ===
# Reshape input to 2D for sklearn
X_cyl = curv_cylinder.reshape(-1, 1)
y_cyl = mean_cylinder

X_sph = curv_sphere.reshape(-1, 1)
y_sph = mean_sphere

# Fit models
model_cyl = LinearRegression().fit(X_cyl, y_cyl)
model_sph = LinearRegression().fit(X_sph, y_sph)

# Predict for a smooth line
x_pred = np.linspace(0, 90, 200).reshape(-1, 1)
y_pred_cyl = model_cyl.predict(x_pred)
y_pred_sph = model_sph.predict(x_pred)

# === Plot ===
plt.figure(figsize=(6, 5))

# Original data points
plt.plot(curv_cylinder, mean_cylinder, 'o', label='Cylinder (data)', color='blue')
plt.plot(curv_sphere, mean_sphere, 's', label='Sphere (data)', color='orange')

# Regression lines
plt.plot(x_pred, y_pred_cyl, '--', label='Cylinder (linear fit)', color='blue')
plt.plot(x_pred, y_pred_sph, '--', label='Sphere (linear fit)', color='orange')

# Axes and labels
plt.xlabel('Curvature')
plt.ylabel('Sensor Reading')
plt.ylim(700, 780)
plt.xlim(0, 90)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Print equations ===
print(f"Cylinder model: Sensor = {model_cyl.coef_[0]:.4f} * Curvature + {model_cyl.intercept_:.4f}")
print(f"Sphere model:   Sensor = {model_sph.coef_[0]:.4f} * Curvature + {model_sph.intercept_:.4f}")
