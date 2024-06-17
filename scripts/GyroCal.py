import pandas as pd
import numpy as np

def gyro_calibration():
  df_raw_data = pd.read_csv('data/acc_gyro_cal_data.csv')
  gyro_data = df_raw_data[['gyro_x', 'gyro_y', 'gyro_z', 'label']]

  offset_x = gyro_data['gyro_x'].sum()/len(gyro_data)
  offset_y = gyro_data['gyro_y'].sum()/len(gyro_data)
  offset_z = gyro_data['gyro_z'].sum()/len(gyro_data)
  gyro_b = np.array([offset_x, offset_y, offset_z])
  print(f"gyro_b = np.array([{offset_x}, {offset_y}, {offset_z}])")

  return gyro_b

gyro_calibration()
