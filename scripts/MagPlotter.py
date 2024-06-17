import pandas as pd
import dataRead as dr
import matplotlib.pyplot as plt
import keyboard as kb
import time

start_time = time.time()
print(start_time)

saving = False  

plt.ion()
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
sc_saved = ax.scatter([], [], [], color="green", label='Saved Data')
sc_unsaved = ax.scatter([], [], [], color="red", label='Unsaved Data', alpha=0.5)
df_raw_data_no_save = pd.DataFrame(columns=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z', 'time'])
df_raw_data_save = pd.DataFrame(columns=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z', 'time'])

ax.set_xlim([-500, 500])
ax.set_ylim([-500, 500])
ax.set_zlim([-500, 500])
ax.set_box_aspect([1, 1, 1])

while True:
  ser_raw_data = dr.readSerial()
  if kb.is_pressed('q'):
    break

  if ser_raw_data is not None and isinstance(ser_raw_data, list) and len(ser_raw_data) == 9:
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
      "time": [add_time]
    })
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
