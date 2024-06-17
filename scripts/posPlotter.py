# --- Libraries ---
import numpy as np
import matplotlib.pyplot as plt
import keyboard as kb
import dataRead as dr  # Assuming dataRead is a custom module for reading serial data
import time
import pandas as pd
from ahrs.filters import Madgwick

# --- Constants ---
# Start time
start_time = time.time()

# AHRS Filter
madgwick = Madgwick()

# Accelerometer Calibration
acc_b = np.array([0.020612244897959153, -0.004531249999999987, 0.019610917537746853])
acc_A_1 = np.array([[0.9993881297164999, 0, 0],
                    [0, 0.9954891896095817, 0],
                    [0, 0, 0.9878554587332272]])

# Gyroscope Calibration
gyro_b = np.array([-1.4368016194331983, 1.4903643724696356, -1.3609311740890693])

# Magnetometer Calibration
mag_b = np.array([79.90798583281772, 81.17560183293999, -450.41810929308957])
mag_A_1 = np.array([[2.2101349783176985, -0.018350529790350026, -0.011089820465294751],
                    [-0.018350529790350092, 2.1568647254581763, 0.020187152925131777],
                    [-0.011089820465294683, 0.020187152925131735, 2.1833925437724604]])

# Saving data flag
saving = True

# --- Dataframe for raw data ---
df_raw_data = pd.DataFrame(columns=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z', 'time'])

# --- Data Acquisition Loop ---
while saving:
  ser_raw_data = dr.readSerial()
  print(ser_raw_data)
  if kb.is_pressed('c'):
    saving = False
  if ser_raw_data is not None and isinstance(ser_raw_data, list) and len(ser_raw_data) == 10:
    df_raw_data = df_raw_data._append(pd.Series(ser_raw_data, index=df_raw_data.columns), ignore_index=True)
  elif ser_raw_data is None:
    df_raw_data = None

print(df_raw_data)

# --- Calibration ---
if df_raw_data is not None:
  acc_data_raw = np.array(df_raw_data[['acc_x', 'acc_y', 'acc_z']])
  acc_data_cal = np.dot(acc_data_raw - acc_b, acc_A_1)
  gyro_data_raw = np.array(df_raw_data[['gyro_x', 'gyro_y', 'gyro_z']])
  gyro_data_cal = gyro_data_raw - gyro_b
  mag_data_raw = np.array(df_raw_data[['mag_x', 'mag_y', 'mag_z']])
  mag_data_cal = np.dot(mag_data_raw - mag_b, mag_A_1)  # Changed from acc_b to mag_b

  # --- AHRS Filtering ---
  Q = np.tile([1., 0., 0., 0.], (len(acc_data_cal), 1)) 
  for i in range(1, len(acc_data_cal)):
    Q[i] = madgwick.updateMARG(Q[i-1], gyr=gyro_data_cal[i], acc=acc_data_cal[i], mag=mag_data_cal[i])

  print(Q)
  
  def euler_from_quaternion(quaternion):
    w, x, y, z = quaternion
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.rad2deg(np.arctan2(t0, t1))
  
    t2 = 2.0 * (w * y - z * x)
    t2 = 1.0 if t2 > +1.0 else t2
    t2 = 1.0 if t2 < -1.0 else t2
    pitch = np.rad2deg(np.arcsin(t2))
  
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.rad2deg(np.arctan2(t3, t4))
  
    return roll, pitch, yaw 
  
  for i in range(len(Q)):
    print(euler_from_quaternion(Q[i]))


