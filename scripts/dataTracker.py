import numpy as np  
import pandas as pd 
import serial  
import serial.tools.list_ports
import keyboard as kb
import time as time

gyro_b = np.array([16.65063346023689,5.123359771573603,0.2288758460236887])
acc_b = np.array([0.016235308313439945,0.01592428077039082,-0.10511789279394046])
acc_A_1 = np.array([[0.9972221442010574,0,0],
                    [0,0.9970994242195597,0],
                    [0,0,0.9880121143442029]])
mag_b = np.array([354.99101508502997,68.63896524827186,458.1190555872505])
mag_A_1 = np.array([[2.3148232680896945,-0.09722382426008001,-0.14101312997607546],
                    [-0.09722382426008014,2.293529669948388,0.05827894371560589],
                    [-0.1410131299760754,0.058278943715605866,2.345025081355761]])      

def get_highest_com_port(): 
  ports = list(serial.tools.list_ports.comports())
  if not ports:                 
    raise Exception("No COM ports found")
  highest_port = max(ports, key=lambda port: int(port.device[3:]))
  return highest_port.device

def readSerial(ser):
  try:
    line = ser.readline()
    decoded_line_raw = line.decode('utf-8').rstrip()
    raw_data = [float(part) for part in decoded_line_raw.split(",")]
    if len(raw_data) == 10:
      return raw_data
    else:
      print("Unexpected data length:", len(raw_data))
      return None
  except Exception as e:
    print("Error reading serial data:", e)
    return None

def datalogger(ser):
  ser.reset_input_buffer()
  saving = True
  collected_data = []
  df_raw_data = pd.DataFrame(columns=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z', 'time'])
  while saving:
    ser_raw_data = readSerial(ser)
    if kb.is_pressed('space'):
      saving = False
    else:
      if ser_raw_data is not None and isinstance(ser_raw_data, list) and len(ser_raw_data) == 10:
        print(ser_raw_data)
        collected_data.append(ser_raw_data)

  df_raw_data = pd.DataFrame(collected_data, columns=df_raw_data.columns)
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
  
  return df_cal_data    

if __name__ == '__main__':
  try:
    highest_com_port = get_highest_com_port()
    ser = serial.Serial(port=highest_com_port, baudrate=460800, timeout=1)
    df_raw_data = datalogger(ser)  
    if not df_raw_data.empty:
      df_cal_data = dataCal(df_raw_data)
      print(df_cal_data)
      if not df_cal_data.empty:
        filename = f"track_data_{int(time.time())}"
        df_cal_data.to_csv(f"data/tracking_data.csv", index=False)
        print(f'data = numpy.genfromtxt("data/{filename}.csv", delimiter=",", skip_header=1)')
      else:
        print("No calibrated data to save.")
    else:
      print("No data collected.")
  except Exception as e:
    print(f"An error occurred: {e}")
        
         