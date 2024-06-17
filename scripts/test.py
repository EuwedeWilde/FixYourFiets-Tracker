
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dataRead as dr
import keyboard as kb
import time

def fit_data(data, b, A_1):
  xm_off  = data[0,0]-b[0,0]
  ym_off  = data[0,1]-b[1,0]
  zm_off  = data[0,2]-b[2,0]
  xm_cal = xm_off *  A_1[0,0] + ym_off *  A_1[0,1]  + zm_off *  A_1[0,2] 
  ym_cal = xm_off *  A_1[1,0] + ym_off *  A_1[1,1]  + zm_off *  A_1[1,2] 
  zm_cal = xm_off *  A_1[2,0] + ym_off *  A_1[2,1]  + zm_off *  A_1[2,2] 
  return np.array([xm_cal, ym_cal, zm_cal])

start_time = time.time()
print(start_time)

saving = False  

b = np.array([[79.90798583],
              [81.17560183],
              [-450.41810929]])
A_1 = np.array([[2.2101349783176985,-0.018350529790350026,-0.011089820465294751],
                [-0.018350529790350092,2.1568647254581763,0.020187152925131777],
                [-0.011089820465294683,0.020187152925131735,2.1833925437724604]])



plt.ion()
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
sc_saved = ax.scatter([], [], [], color="green", label='Saved Data')
sc_unsaved = ax.scatter([], [], [], color="red", label='Unsaved Data', alpha=0.5)
df_raw_data_no_save = pd.DataFrame(columns=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z', 'time'])
df_raw_data_save = pd.DataFrame(columns=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z', 'time'])
u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
ax.plot_surface(x*1000, y*1000, z*1000, color='blue', alpha=0.5, label='Perfect Fit')
ax.set_box_aspect([1,1,1])

ax.set_xlim([-1200, 1200])
ax.set_ylim([-1200, 1200])
ax.set_zlim([-1200, 1200])
ax.set_box_aspect([1, 1, 1])

while True:
  ser_raw_data = dr.readSerial()
  if kb.is_pressed('q'):
    break

  if ser_raw_data is not None and isinstance(ser_raw_data, list) and len(ser_raw_data) == 10:
    add_time = time.time() - start_time
    new_row = pd.DataFrame({
      "acc_x": [ser_raw_data[0]],
      "acc_y": [ser_raw_data[1]],
      "acc_z": [ser_raw_data[2]],
      "gyro_x": [ser_raw_data[3]],
      "gyro_y": [ser_raw_data[4]],
      "gyro_z": [ser_raw_data[5]],
      "mag_x": [ser_raw_data[6]],
      "mag_y": [ser_raw_data[7]],
      "mag_z": [ser_raw_data[8]],
      "time": [ser_raw_data[9]]
    })
    mag_data_raw = np.array(new_row[['mag_x', 'mag_y', 'mag_z']])
    mag_data_calibrated = fit_data(mag_data_raw, b, A_1)
    new_row['mag_x'] = mag_data_calibrated[0]
    new_row['mag_y'] = mag_data_calibrated[1]
    new_row['mag_z'] = mag_data_calibrated[2]
    if kb.is_pressed('p') and saving == False:
      saving = True
      print("Saving now..")
    elif kb.is_pressed('o') and saving == True:
      saving = False
      print("Stopped saving...")
    elif kb.is_pressed('c'):
      df_raw_data_save.to_csv("data/mag_cal_data.csv", index=False)
      print("Saved as: data/mag_cal_data.csv")
      break

    if saving:
      if df_raw_data_save.empty:
        df_raw_data_save = new_row
      else:
        df_raw_data_save = pd.concat([df_raw_data_save, new_row], ignore_index=True)
    else:
      if df_raw_data_no_save.empty:
        df_raw_data_no_save = new_row
      else:
        df_raw_data_no_save = pd.concat([df_raw_data_no_save, new_row], ignore_index=True)

    sc_saved._offsets3d = (df_raw_data_save["mag_x"], df_raw_data_save["mag_y"], df_raw_data_save["mag_z"])
    sc_unsaved._offsets3d = (df_raw_data_no_save["mag_x"], df_raw_data_no_save["mag_y"], df_raw_data_no_save["mag_z"])
    plt.draw()
    plt.pause(0.01)

plt.ioff()
plt.show()