import numpy as np  
import pandas as pd 
import serial  
import serial.tools.list_ports
import keyboard as kb
import time as time

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

if __name__ == '__main__':
  try:
    highest_com_port = get_highest_com_port()
    ser = serial.Serial(port=highest_com_port, baudrate=460800, timeout=1)
    df_raw_data = datalogger(ser)  
    if not df_raw_data.empty:
      filename = f"track_data_{int(time.time())}"
      df_raw_data.to_csv(f"data/tracking_data.csv", index=False)
      print(f'data = numpy.genfromtxt("data/{filename}.csv", delimiter=",", skip_header=1)')
    else:
      print("No data collected.")
  except Exception as e:
    print(f"An error occurred: {e}")
        
         