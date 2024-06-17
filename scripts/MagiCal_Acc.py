import numpy as np
from scipy.optimize import least_squares
import pandas as pd
import matplotlib.pyplot as plt

df_raw_data = pd.read_csv('data/acc_cal_data.csv')
acc_data_raw = np.array(df_raw_data[['acc_x', 'acc_y', 'acc_z']])

def rotation_matrix(data, params):
  xc, yc, zc = params[0:3]
  alpha, beta, gamma = params[6:9]
  Rx = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
  Ry = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
  Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
  R = Rx @ Ry @ Rz
  data_rot = np.dot(data - np.array([xc, yc, zc]), R.T)
  return data_rot

def ellipsoid_func(params, data):
  a, b, c = params[3:6]
  data_rot = rotation_matrix(data, params)
  x, y, z = data_rot.T
  return (x / a)**2 + (y / b)**2 + (z / c)**2 - 1

def fit_ellipsoid(data):
  x0 = np.r_[np.mean(data, axis=0), 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
  result = least_squares(ellipsoid_func, x0, args=(data,))
  return result.x

def calibrate_data(data, params):
  a, b, c = params[3:6]
  data_rot = rotation_matrix(data, params)
  data_scaled = np.array([data_rot[:, 0] / a, data_rot[:, 1] / b, data_rot[:, 2] / c]).T
  return data_scaled

params = fit_ellipsoid(acc_data_raw)
print("Ellipsoid Parameters:")
print(params)

acc_data_calibrated = calibrate_data(acc_data_raw, params)

print("Calibrated accnetometer Data:")
print(acc_data_calibrated)

fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection="3d")
ax.scatter3D(acc_data_raw[:, 0], acc_data_raw[:, 1], acc_data_raw[:, 2], color="red", label='Unfitted Data')
ax.scatter3D(acc_data_calibrated[:, 0], acc_data_calibrated[:, 1], acc_data_calibrated[:, 2], color="green", label='Fitted Data')
u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
ax.plot_surface(x*400, y*400, z*400, color='blue', alpha=0.5, label='Perfect Fit')
ax.set_box_aspect([1,1,1])
plt.legend()
plt.title("Fitted Ellipsoid")
plt.show()
