import serial

ser = serial.Serial(port='COM12', baudrate=115200, timeout=1)
decoded_line = None

def readSerial():
  try:
    line = ser.readline()
    decoded_line_raw = line.decode('utf-8').rstrip()
    decoded_line = decoded_line_raw
    raw_data = [float(part) for part in decoded_line.split(",")]
    return raw_data
  except:
    print("Error reading serial data")

if __name__ == '__main__':
  while True:
    ser_raw_data = readSerial()
    if ser_raw_data:
      print(ser_raw_data)
    else:
      print("No data")