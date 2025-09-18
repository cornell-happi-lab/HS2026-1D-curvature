from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

# === Step 1: Aggregate all data ===
def build_df(avg_ft, avg_sensor, curvature):
    return pd.DataFrame({
        'Force': avg_ft.values,
        'Sensor': avg_sensor.values,
        'Curvature': curvature
    })

df = pd.read_csv("calibration_data/sensor5_cy_0.csv")
df['FT_Z'] = -df['FT_Z']
df['block'] = df['S5'] + df['S6'] + df['S9'] + df['S10']
grouped = df.groupby('Step')
avg_ft = grouped['FT_Z'].mean()
avg_sensor = grouped['block'].mean() 
std_sensor = grouped['block'].std()

df1 = pd.read_csv("calibration_data/sensor5_cy_20.csv")
df1['FT_Z'] = -df1['FT_Z']
df1['block'] = df1['S5'] + df1['S6'] + df1['S9'] + df1['S10']
grouped1 = df1.groupby('Step')
avg_ft1 = grouped1['FT_Z'].mean()
avg_sensor1 = grouped1['block'].mean() 
std_sensor1 = grouped1['block'].std()

df2 = pd.read_csv("calibration_data/sensor5_cy_40.csv")
df2['FT_Z'] = -df2['FT_Z']
df2['block'] = df2['S5'] + df2['S6'] + df2['S9'] + df2['S10']
grouped2 = df2.groupby('Step')
avg_ft2 = grouped2['FT_Z'].mean()
avg_sensor2 = grouped2['block'].mean() 
std_sensor2 = grouped2['block'].std()

df3 = pd.read_csv("calibration_data/sensor5_cy_60.csv")
df3['FT_Z'] = -df3['FT_Z']
df3['block'] = df3['S5'] + df3['S6'] + df3['S9'] + df3['S10']
grouped3 = df3.groupby('Step')
avg_ft3 = grouped3['FT_Z'].mean()
avg_sensor3 = grouped3['block'].mean() 
std_sensor3 = grouped3['block'].std()

df4 = pd.read_csv("calibration_data/sensor5_cy_80.csv")
df4['FT_Z'] = -df4['FT_Z']
df4['block'] = df4['S5'] + df4['S6'] + df4['S9'] + df4['S10']
grouped4 = df4.groupby('Step')
avg_ft4 = grouped4['FT_Z'].mean()
avg_sensor4 = grouped4['block'].mean() 
std_sensor4 = grouped4['block'].std()

# df5 = pd.read_csv("calibration_data/sensor5_cy_100.csv")
# df5['FT_Z'] = -df5['FT_Z']
# df5['block'] = df5['S5'] + df5['S6'] + df5['S9'] + df5['S10']
# grouped5 = df5.groupby('Step')
# avg_ft5 = grouped5['FT_Z'].mean()
# avg_sensor5 = grouped5['block'].mean() 
# std_sensor5 = grouped5['block'].std()


dfs = []
dfs.append(build_df(avg_ft, avg_sensor, curvature=0))
dfs.append(build_df(avg_ft1, avg_sensor1, curvature=20))
dfs.append(build_df(avg_ft2, avg_sensor2, curvature=40))
dfs.append(build_df(avg_ft3, avg_sensor3, curvature=60))
dfs.append(build_df(avg_ft4, avg_sensor4, curvature=80))
# dfs.append(build_df(avg_ft5, avg_sensor5, curvature=100))

df_all = pd.concat(dfs, ignore_index=True)

# === Step 2: Create the model ===
X = df_all[['Sensor', 'Curvature']]
y = df_all['Force']

# Use polynomial regression
model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
model.fit(X, y)

# Print out the equation
poly = model.named_steps['polynomialfeatures']
lin_reg = model.named_steps['linearregression']
input_feature_names = X.columns  # ['Sensor', 'Curvature']
poly_feature_names = poly.get_feature_names_out(input_feature_names)
coefficients = lin_reg.coef_
intercept = lin_reg.intercept_
terms = [f"{coef:.6f}*{name}" for coef, name in zip(coefficients, poly_feature_names)]
equation = " + ".join(terms)
equation = f"Predicted Force = {equation} + {intercept:.6f}"
print(equation)

# === Step 3: Evaluate ===
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print(f"Model R² Score: {r2:.4f}")

# === Step 4: Example usage ===
# Predict force given sensor=500 and curvature=60
example_force = model.predict([[500, 60]])
print(f"Predicted force for sensor=500, curvature=60: {example_force[0]:.2f} N")
baseline = model.predict([[0, 0]])
print(f"Predicted force for sensor=0, curvature=0: {baseline[0]:.2f} N")

# === Step 5: 3D Plot (optional) ===
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter actual data
ax.scatter(df_all['Sensor'], df_all['Curvature'], y, label='True', c='blue', alpha=0.5)

# Surface plot
sensor_range = np.linspace(df_all['Sensor'].min(), df_all['Sensor'].max(), 30)
curv_range = np.linspace(df_all['Curvature'].min(), df_all['Curvature'].max(), 30)
SENSOR, CURV = np.meshgrid(sensor_range, curv_range)
X_grid = np.c_[SENSOR.ravel(), CURV.ravel()]
Z = model.predict(X_grid).reshape(SENSOR.shape)

ax.plot_surface(SENSOR, CURV, Z, alpha=0.6, color='orange', edgecolor='none')

ax.set_xlabel('Sensor Readings')
ax.set_ylabel('Curvature (1/m)')
ax.set_zlabel('FT Z Force (N)')

# Define ranges
ax.set_xlim(0, 1200)   # X-axis (Sensor Readings)
ax.set_ylim(0, 80)     # Y-axis (Curvature)
ax.set_zlim(0, 30)     # Z-axis (Force)


ax.set_title('Model: Force ~ Sensor + Curvature')
plt.tight_layout()
plt.show()
