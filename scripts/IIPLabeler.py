import imufusion as imu
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import pandas as pd
import arff

gyro_b = np.array([16.65063346023689,5.123359771573603,0.2288758460236887])
acc_b = np.array([0.016235308313439945,0.01592428077039082,-0.10511789279394046])
acc_A_1 = np.array([[0.9972221442010574,0,0],
                    [0,0.9970994242195597,0],
                    [0,0,0.9880121143442029]])
mag_b = np.array([354.99101508502997,68.63896524827186,458.1190555872505])
mag_A_1 = np.array([[2.3148232680896945,-0.09722382426008001,-0.14101312997607546],
                    [-0.09722382426008014,2.293529669948388,0.05827894371560589],
                    [-0.1410131299760754,0.058278943715605866,2.345025081355761]])      

def dataCal(df_raw_data):
  df_raw_data = df_raw_data.iloc[5:] 
  if df_raw_data.empty:
    print("Not enough data to process after removing the first 5 rows.")
    return pd.DataFrame()

  start_time = df_raw_data['time'].iloc[0]
  df_cal_data = df_raw_data.copy()
  
  acc_x = df_raw_data['acc_x'].to_numpy()
  acc_y = df_raw_data['acc_y'].to_numpy()
  acc_z = df_raw_data['acc_z'].to_numpy()
  
  df_cal_data['acc_x'] = (acc_x * acc_A_1[0, 0] - acc_b[0])
  df_cal_data['acc_y'] = (acc_y * acc_A_1[1, 1] - acc_b[1])
  df_cal_data['acc_z'] = (acc_z * acc_A_1[2, 2] - acc_b[2])

  df_cal_data[['gyro_x', 'gyro_y', 'gyro_z']] = df_raw_data[['gyro_x', 'gyro_y', 'gyro_z']] - gyro_b
   
  # Magnetometer calibration
  mag_raw = df_raw_data[['mag_x', 'mag_y', 'mag_z']].to_numpy()
  mag_cal = (mag_A_1 @ (mag_raw - mag_b).T).T
  df_cal_data[['mag_x', 'mag_y', 'mag_z']] = mag_cal
  
  # Adjust time
  df_cal_data['time'] = (df_raw_data['time'] - start_time) / 1000000
  
  return df_cal_data    

class DataLabeler:
  def __init__(self, ax, timestamp, raw_acceleration, calculated_acceleration):
    self.ax = ax
    self.timestamp = timestamp
    self.raw_acceleration = raw_acceleration
    self.calculated_acceleration = calculated_acceleration
    self.selected_indices = set()
    self.span = SpanSelector(ax, self.onselect, 'horizontal', useblit=True, interactive=True,
                             props=dict(alpha=0.5, facecolor='red'))

  def onselect(self, xmin, xmax):
    indmin, indmax = np.searchsorted(self.timestamp, (xmin, xmax))
    indmax = min(len(self.timestamp) - 1, indmax)
    self.selected_indices.update(range(indmin, indmax + 1))
    self.ax.axvspan(xmin, xmax, facecolor='red', alpha=0.5)
    plt.draw()

  def save_labeled_data(self, filename):
    labels = ['B' if i in self.selected_indices else 'A' for i in range(len(self.timestamp))]
    df = pd.DataFrame({
      'timestamp': self.timestamp,
      'ACC_x': self.raw_acceleration[:, 0],
      'ACC_y': self.raw_acceleration[:, 1],
      'ACC_z': self.raw_acceleration[:, 2],
      'label': labels
    })

    arff_data = {
      'description': '',
      'relation': 'pen_movement',
      'attributes': [
        ('timestamp', 'NUMERIC'),
        ('ACC_x', 'REAL'),
        ('ACC_y', 'REAL'),
        ('ACC_z', 'REAL'),
        ('label', ['A', 'B'])
      ],
      'data': df.values.tolist()
    }
    
    with open(filename, 'w') as f:
      arff.dump(arff_data, f)
    print(f"Labeled data saved to {filename}")

def main():
  data = np.genfromtxt("data/tracking_data.csv", delimiter=",", skip_header=1)
  data_cal = dataCal(data)
  sample_rate = 1000  
  timestamp = data[:, 9]
  gyroscope = data[:, 3:6]
  accelerometer = data[:, 0:3]  # Raw accelerometer data
  magnetometer = data[:, 6:9] 

  offset = imu.Offset(sample_rate)
  ahrs = imu.Ahrs()

  ahrs.settings = imu.Settings(imu.CONVENTION_NWU,
                              0.5,  # gain
                              2000,  # gyroscope range
                              10,  # acceleration rejection
                              500,  # magnetic rejection (increase for higher reliance on magnetometer)
                              5 * sample_rate)  # rejection timeout = 5 seconds

  delta_time = np.diff(timestamp, prepend=timestamp[0])
  euler = np.empty((len(timestamp), 3))
  internal_states = np.empty((len(timestamp), 6))
  flags = np.empty((len(timestamp), 4))
  calculated_acceleration = np.empty((len(timestamp), 3))

  for index in range(len(timestamp)):
    gyroscope[index] = offset.update(gyroscope[index])
    ahrs.update(gyroscope[index], accelerometer[index], magnetometer[index], delta_time[index])
    euler[index] = ahrs.quaternion.to_euler()
    ahrs_internal_states = ahrs.internal_states
    internal_states[index] = np.array([ahrs_internal_states.acceleration_error,
                                      ahrs_internal_states.accelerometer_ignored,
                                      ahrs_internal_states.acceleration_recovery_trigger,
                                      ahrs_internal_states.magnetic_error,
                                      ahrs_internal_states.magnetometer_ignored,
                                      ahrs_internal_states.magnetic_recovery_trigger])
    ahrs_flags = ahrs.flags
    flags[index] = np.array([ahrs_flags.initialising,
                            ahrs_flags.angular_rate_recovery,
                            ahrs_flags.acceleration_recovery,
                            ahrs_flags.magnetic_recovery])
    calculated_acceleration[index] = 9.81 * ahrs.earth_acceleration

  print(f"Calculated ACC: {calculated_acceleration}")
  fig, ax = plt.subplots(1, 1, figsize=(12, 6))

  calculated_acceleration_x = calculated_acceleration[:, 0]
  calculated_acceleration_y = calculated_acceleration[:, 1]
  calculated_acceleration_z = calculated_acceleration[:, 2]

  ax.plot(timestamp, calculated_acceleration_x, label='Calculated Acceleration X')
  ax.plot(timestamp, calculated_acceleration_y, label='Calculated Acceleration Y')
  ax.plot(timestamp, calculated_acceleration_z, label='Calculated Acceleration Z')
  ax.set_title('Calculated Acceleration')
  ax.legend()

  labeler = DataLabeler(ax, timestamp, accelerometer, calculated_acceleration)
  plt.show()
  labeler.save_labeled_data("data/tracking_data_labeled.arff")

if __name__ == "__main__":
  main()
