import time
import keyboard as kb
import serial
import serial.tools.list_ports
import pandas as pd

def get_highest_com_port():
  ports = list(serial.tools.list_ports.comports())
  if not ports:
    raise Exception("No COM ports found")
  highest_port = max(ports, key=lambda port: int(port.device[3:]))  # Assumes COM port names are in the format 'COMx'
  return highest_port.device

def countdown(n):
  for i in range(n, 0, -1):
    print(i)
    time.sleep(1)

def readSerial(ser):
  try:
    line = ser.readline()
    decoded_line_raw = line.decode('utf-8').rstrip()
    decoded_line = decoded_line_raw
    raw_data = [float(part) for part in decoded_line.split(",")]
    return raw_data
  except Exception as e:
    print("Error reading serial data:", e)
    return None

def datalogger(ser, sensor, dir):
  ser.reset_input_buffer()
  start_time = int(time.time())
  saving = True
  collected_data = []
  df_raw_data = pd.DataFrame(columns=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z', 'time', 'sensor', 'dir'])

  if sensor == 'gyro':
    measure_time = 10
  elif sensor == 'acc':
    measure_time = 5  
  elif sensor == 'mag':
    measure_time = 0

  while saving:
    ser_raw_data = readSerial(ser)
    if time.time() - start_time >= measure_time and sensor != "mag":
      saving = False
    elif sensor == "mag" and kb.is_pressed('space'):
      saving = False
    else:
      if ser_raw_data is not None and isinstance(ser_raw_data, list) and len(ser_raw_data) == 10:
        ser_raw_data.extend([sensor, dir])
        print(ser_raw_data)
        collected_data.append(ser_raw_data)

  df_raw_data = pd.DataFrame(collected_data, columns=df_raw_data.columns)
  return df_raw_data

def gyroscope_calibration(ser):
  print("We are going to calibrate the gyroscope. During this process, please make sure the IMU is placed on a flat surface and it is not moving for 10 seconds.\nWhen you are ready press Spacebar.")
  kb.wait('space')
  print("Calibrating gyroscope in:")
  time.sleep(1)
  countdown(3)
  print("Calibrating gyroscope...")
  df_raw_data_gyro = datalogger(ser, "gyro", None)
  print("Calibration process finished.")
  return df_raw_data_gyro

def accelerometer_calibration(ser):
  directions = ['x+', 'x-', 'y+', 'y-', 'z+', 'z-']
  print("We are going to calibrate the accelerometer. During this process, we are going to measure 6 axis. Each time, make sure the axis perpendicular to the ground and not moving for 5 seconds.\nWhen you are ready press Spacebar.")
  kb.wait('space')
  df_raw_data_acc = None
  for i in range(6):
    print(f"Calibrating accelerometer {directions[i]}, press Spacebar when ready.")
    kb.wait('space')
    time.sleep(1)
    countdown(3)
    print(f"Calibrating accelerometer {directions[i]}...")
    df_raw_data_acc_new = datalogger(ser, "acc", directions[i])
    if df_raw_data_acc is None:
      df_raw_data_acc = df_raw_data_acc_new
    else:
      df_raw_data_acc = pd.concat([df_raw_data_acc, df_raw_data_acc_new], ignore_index=True)
  print("Calibration process finished.")
  return df_raw_data_acc  

def magnetometer_calibration(ser):
  print("We are going to calibrate the magnetometer. During this process, move and rotate the IMU in all directions, at least at 1 meter distance from any metals, do this until you see a nice ellipsoid.\nWhen you are ready press Spacebar.")
  kb.wait('space')
  print("Calibrating magnetometer in:")
  time.sleep(1)
  countdown(3)
  print("Calibrating magnetometer...")
  df_raw_data_mag = datalogger(ser, "mag", None)
  print("Calibration process finished.")
  return df_raw_data_mag

if __name__ == "__main__":
  try:
    highest_com_port = get_highest_com_port()
    ser = serial.Serial(port=highest_com_port, baudrate=460800, timeout=1)
    print("Starting calibration process...")
    time.sleep(1)
    df_raw_data_gyro = gyroscope_calibration(ser)
    time.sleep(1)
    df_raw_data_acc = accelerometer_calibration(ser)
    time.sleep(1) 
    df_raw_data_mag = magnetometer_calibration(ser)
    ser.close() 
    print("Complete calibration process finished.")
    df_raw_data = pd.concat([df_raw_data_gyro, df_raw_data_acc, df_raw_data_mag], ignore_index=True)
    df_raw_data.to_csv(f"data/cal_data.csv", index=False)
    print(f"Saved as: data/cal_data_{int(time.time())}.csv")
  except Exception as e:
    print(f"An error occurred: {e}")
 