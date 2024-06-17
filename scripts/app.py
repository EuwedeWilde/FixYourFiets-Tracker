import numpy as np
import pandas as pd
import serial
import time
import imufusion as imu
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import dataProcess as dp
import keyboard 

processing = False
measuring = False

ser = serial.Serial(port='COM29', baudrate=115200, timeout=1)
decoded_line = None

gyro_b = np.array([16.92060060891226,5.1918737890949345,-0.013418212012178245])
acc_b = np.array([0.015384318158501886,0.013868609915495933,-0.10010740405861607])
acc_A_1 = np.array([[0.9954217416240858,0,0],
                    [0,0.9964198442047845,0],
                    [0,0,0.9901622740197321]])
mag_b = np.array([352.56488147776037,103.94996908802119,396.33904893520975])
mag_A_1 = np.array([[2.146488412990616,-0.03207225304387284,-0.09541198885352041],
                    [-0.03207225304387285,2.172763839773808,0.016533800986287636],
                    [-0.0954119888535202,0.016533800986287796,2.2294967732136968]])

def readSerial():
  try:
    line = ser.readline()
    decoded_line_raw = line.decode('utf-8').rstrip()
    raw_data = [float(part) for part in decoded_line_raw.split(",")]
    if len(raw_data) == 10:
      return raw_data
    else:
      return None
  except Exception as e:
    return None

def datalogger():
  global measuring, processing
  ser.reset_input_buffer()
  collected_data = []
  collected_measuring_data = []
  if keyboard.is_pressed('space'):
    collected_data = []
  rolling_buffer = []  # Buffer to keep the last 800 rows
  buffer_size = 800
  df_raw_data = pd.DataFrame(columns=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z', 'time'])

  measuring_start_time = None  # Timer for measuring
  non_measuring_start_time = None  # Timer for non-measuring
  
  while True:
    ser_raw_data = readSerial()
    if ser_raw_data is not None and isinstance(ser_raw_data, list) and len(ser_raw_data) == 10:
      print(ser_raw_data)
      collected_data.append(ser_raw_data)
      rolling_buffer.append(ser_raw_data)
      
      # Maintain the rolling buffer size
      if len(rolling_buffer) > buffer_size:
        rolling_buffer.pop(0)
      
      acc_x, acc_y, acc_z = ser_raw_data[:3]
      acc_tot = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
      
      if (acc_tot > 1.5):
        if not measuring:
          measuring = True
          start_time = ser_raw_data[9]
          collected_measuring_data.extend(rolling_buffer)  # Include the first 800 rows before measurement start
        non_measuring_start_time = None  # Reset non-measuring timer
      else:
        if measuring:
          if non_measuring_start_time is None:
            non_measuring_start_time = time.time()
          elif time.time() - non_measuring_start_time >= 5:
            measuring = False
            processing = True
            end_time = ser_raw_data[9]
            # Include the rows between start_time and end_time
            start_index = next(i for i, row in enumerate(collected_data) if row[9] >= start_time)
            end_index = next(i for i, row in enumerate(collected_data) if row[9] >= end_time)
            collected_measuring_data.extend(collected_data[start_index:end_index+1])
            time.sleep(0.5)
            measuring_start_time = None  # Reset measuring timer
            break

  df_raw_data = pd.DataFrame(collected_measuring_data, columns=df_raw_data.columns)
  return df_raw_data

def dataCal(df_raw_data):
  df_raw_data = df_raw_data.iloc[5:]  # Remove the first 5 rows
  if df_raw_data.empty:
    print("Not enough data to process after removing the first 5 rows.")
    return pd.DataFrame()

  start_time = df_raw_data['time'].iloc[0]
  df_cal_data = df_raw_data.copy()
  
  # Convert Series to numpy arrays
  acc_x = df_raw_data['acc_x'].to_numpy()
  acc_y = df_raw_data['acc_y'].to_numpy()
  acc_z = df_raw_data['acc_z'].to_numpy()
  
  # Apply calibration
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
  print(df_cal_data)
  return df_cal_data

if __name__ == '__main__':
  while True:
    if not processing:
      df_raw_data = datalogger()
    elif processing:
      if not df_raw_data.empty:
        df_cal_data = dataCal(df_raw_data)
        if not df_cal_data.empty:
          filename = f"track_data_{int(time.time())}"
          df_cal_data.to_csv(f"data/tracking_data.csv", index=False)
          time.sleep(1)
          dp.main()
          print(f'data = numpy.genfromtxt("data/{filename}.csv", delimiter=",", skip_header=1)')
          processing = False  # Reset processing for the next cycle
        else:
          print("No calibrated data to save.")
      else:
        print("No data collected.")
