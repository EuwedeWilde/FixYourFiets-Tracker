import pandas as pd
import numpy as np

def acc_calibration():
  df_raw_data = pd.read_csv('data/acc_gyro_cal_data.csv')
  acc_data = df_raw_data[['acc_x', 'acc_y', 'acc_z', 'label']]
  num_x_up = acc_data['label'].value_counts()["x up"]
  num_x_down = acc_data['label'].value_counts()["x down"]
  num_y_up = acc_data['label'].value_counts()["y up"]
  num_y_down = acc_data['label'].value_counts()["y down"]
  num_z_up = acc_data['label'].value_counts()["z up"]
  num_z_down = acc_data['label'].value_counts()["z down"]


  offset_x = (acc_data[(acc_data['label'] == 'x up')]['acc_x'].sum() / num_x_up + acc_data[(acc_data['label'] == 'x down')]['acc_x'].sum()  / num_x_down) / 2
  offset_y = (acc_data[(acc_data['label'] == 'y up')]['acc_y'].sum() / num_y_up + acc_data[(acc_data['label'] == 'y down')]['acc_y'].sum()  / num_y_down) / 2
  offset_z = (acc_data[(acc_data['label'] == 'z up')]['acc_z'].sum() / num_z_up + acc_data[(acc_data['label'] == 'z down')]['acc_z'].sum()  / num_z_down) / 2

  scale_x = 2 / abs(acc_data[(acc_data['label'] == 'x up')]['acc_x'].sum() / num_x_up - acc_data[(acc_data['label'] == 'x down')]['acc_x'].sum()  / num_x_down)
  scale_y = 2 / abs(acc_data[(acc_data['label'] == 'y up')]['acc_y'].sum() / num_y_up - acc_data[(acc_data['label'] == 'y down')]['acc_y'].sum()  / num_y_down)
  scale_z = 2 / abs(acc_data[(acc_data['label'] == 'z up')]['acc_z'].sum() / num_z_up - acc_data[(acc_data['label'] == 'z down')]['acc_z'].sum()  / num_z_down)

  acc_b = np.array([offset_x, offset_y, offset_z])
  print(f"acc_b = np.array([{offset_x}, {offset_y}, {offset_z}])")
  acc_A_1 = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, scale_z]])
  print(f"acc_A_1 = np.array([[{scale_x}, 0, 0],\n\t\t    [0, {scale_y}, 0],\n\t\t    [0, 0, {scale_z}]])")

  return acc_b, acc_A_1

acc_calibration()
