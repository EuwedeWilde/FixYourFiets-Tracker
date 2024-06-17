import imufusion as imu
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import seaborn as sns

def rotate_position(position, angle):
  rotation_matrix = np.array([
    [np.cos(angle), -np.sin(angle)],
    [np.sin(angle), np.cos(angle)]
  ])
  
  rotated_position = np.dot(position[:, :2], rotation_matrix.T)
  
  return np.hstack((rotated_position, position[:, 2].reshape(-1, 1)))


def main():
  data = np.genfromtxt("data/tracking_data.csv", delimiter=",", skip_header=1)

  sample_rate = 1000  
  timestamp = data[:, 9]
  gyroscope = data[:, 3:6]
  accelerometer = data[:, 0:3]
  magnetometer = data[:, 6:9] 

  same_start_end = True

  rotation_angle = np.radians(90)

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
  acceleration = np.empty((len(timestamp), 3))

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
    acceleration[index] = 9.81 * ahrs.earth_acceleration


  print(f"ACC: {acceleration}")
  fig, ax = plt.subplots(5, 1, figsize=(9, 9), gridspec_kw={'height_ratios': [6, 6, 6, 1, 1]})

  acceleration_x = acceleration[:, 0]
  acceleration_y = acceleration[:, 1]
  acceleration_z = acceleration[:, 2]

  ax[0].plot(timestamp, acceleration_x, label='Acceleration X')
  ax[0].plot(timestamp, acceleration_y, label='Acceleration Y')
  ax[0].plot(timestamp, acceleration_z, label='Acceleration Z')
  ax[0].set_title('Acceleration')
  ax[0].legend()

  is_moving = np.empty(len(timestamp))

  for index in range(len(timestamp)):
    is_moving[index] = np.sqrt(acceleration[index].dot(acceleration[index])) > 0.5  # threshold = 3 m/s/s

  margin = int(0.1 * sample_rate)  # 100 ms

  for index in range(len(timestamp) - margin):
    is_moving[index] = any(is_moving[index:(index + margin)])  # add leading margin

  for index in range(len(timestamp) - 1, margin, -1):
    is_moving[index] = any(is_moving[(index - margin):index])  # add trailing margin

  velocity = np.zeros((len(timestamp), 3))

  for index in range(1, len(timestamp)):  # Start from 1 to avoid using velocity[-1]
    if is_moving[index]:  # only integrate if moving
      velocity[index] = velocity[index - 1] + delta_time[index] * acceleration[index]

  is_moving_diff = np.diff(is_moving, append=is_moving[-1])

  @dataclass
  class IsMovingPeriod:
    start_index: int = -1
    stop_index: int = -1

  is_moving_periods = []
  is_moving_period = IsMovingPeriod()

  for index in range(len(timestamp)):
    if is_moving_period.start_index == -1:
      if is_moving_diff[index] == 1:
        is_moving_period.start_index = index
    elif is_moving_period.stop_index == -1:
      if is_moving_diff[index] == -1:
        is_moving_period.stop_index = index
        is_moving_periods.append(is_moving_period)
        is_moving_period = IsMovingPeriod()

  num_is_moving = np.sum(is_moving == 1)
  ax[3].plot(timestamp, is_moving, label='Is Moving')
  ax[3].set_title('Is Moving')
  ax[3].legend()

  velocity_drift = np.zeros((len(timestamp), 3))

  for is_moving_period in is_moving_periods:
    start_index = is_moving_period.start_index
    stop_index = is_moving_period.stop_index
    t = [timestamp[start_index], timestamp[stop_index]]
    x = [velocity[start_index, 0], velocity[stop_index, 0]]
    y = [velocity[start_index, 1], velocity[stop_index, 1]]
    z = [velocity[start_index, 2], velocity[stop_index, 2]]  # Corrected z to match the length of x and y
    t_new = timestamp[start_index:(stop_index + 1)]

    if min(t) <= min(t_new) and max(t) >= max(t_new):  # Check if t_new values are within t range
      velocity_drift[start_index:(stop_index + 1), 0] = interp1d(t, x)(t_new)
      velocity_drift[start_index:(stop_index + 1), 1] = interp1d(t, y)(t_new)
      velocity_drift[start_index:(stop_index + 1), 2] = interp1d(t, z)(t_new)
    else:
      print(f"Skipping interpolation for period starting at index {start_index} due to out-of-bounds values.")

  velocity = velocity - velocity_drift
  print(velocity)

  velocity_x = velocity[:, 0]
  velocity_y = velocity[:, 1]
  velocity_z = velocity[:, 2]

  ax[1].plot(timestamp, velocity_x, label='Velocity X')
  ax[1].plot(timestamp, velocity_y, label='Velocity Y')
  ax[1].plot(timestamp, velocity_z, label='Velocity Z')
  ax[1].set_title('Velocity')
  ax[1].legend()

  in_air = np.zeros(len(timestamp), dtype=bool)  # Initialize the in_air array
  in_air_period = 100
  # Calculate velocity changes and set the in_air variable
  for index in range(len(timestamp) - in_air_period):
    velocity_change = np.linalg.norm(velocity[index + in_air_period] - velocity[index])
    if velocity_change > 0.4:
      in_air[index] = True

  position = np.zeros((len(timestamp), 3))

  for index in range(1, len(timestamp)):  
    position[index] = position[index - 1] + delta_time[index] * velocity[index]

  if same_start_end:
    last_position = position[-1]
    position_drift = last_position / num_is_moving
    position_drift_total = np.zeros(3)
    for index in range(len(position)):  
      if is_moving[index] == 1:
        position_drift_total = position_drift_total + position_drift
      position[index] = position[index] - position_drift_total

  print(position)

  position = rotate_position(position, rotation_angle)

  position_x = position[:, 0]
  position_y = position[:, 1]
  position_z = position[:, 2]

  ax[2].plot(timestamp, position_x, label='Position X')
  ax[2].plot(timestamp, position_y, label='Position Y')
  ax[2].plot(timestamp, position_z, label='Position Z')
  ax[2].set_title('Position')
  ax[2].legend()

  max_x_index = np.argmax(position[:, 0])
  min_x_index = np.argmin(position[:, 0])
  max_y_index = np.argmax(position[:, 1])
  min_y_index = np.argmin(position[:, 1])

  pos_max_x = position[max_x_index, 0]
  pos_min_x = position[min_x_index, 0]
  pos_max_y = position[max_y_index, 1]
  pos_min_y = position[min_y_index, 1]
  
  pos_tot_x = abs(pos_max_x - pos_min_x)
  pos_step_x = pos_tot_x / 12
  pos_tot_y = abs(pos_max_y - pos_min_y)
  pos_step_y = pos_tot_y / 12

  heatmap = np.zeros((14, 14))
  for i in range(14):
    for j in range(14):
      heatmap[13-j, i] = np.count_nonzero(
        (position[:, 1] >= pos_min_y + (j-1) * pos_step_y) & 
        (position[:, 1] < pos_min_y + j * pos_step_y) &
        (position[:, 0] >= pos_min_x + (i-1) * pos_step_x) & 
        (position[:, 0] < pos_min_x + i * pos_step_x)  
      )

  heatmap_values = heatmap[heatmap > 0]
  median_value = np.median(heatmap_values)

  heatmap[heatmap <= 0.5 * median_value] = 0
  heatmap[heatmap >= 1.5 * median_value] = 1.5 * median_value

  log_heatmap = np.log10(heatmap + 1)
  max_heatmap = np.max(log_heatmap)
  log_heatmap = log_heatmap / max_heatmap 
  print(log_heatmap)

  plt.figure(figsize=(8, 6))
  ax = sns.heatmap(log_heatmap, linewidth=0.5, vmin=0, vmax=np.log10(np.max(heatmap) + 1))

  print(f"Max position: {pos_max_x}")
  print(f"Min position: {pos_min_x}")
  print(f"Max position: {pos_max_y}")
  print(f"Min position: {pos_min_y}")
  print(f"Total X position: {pos_tot_x}")
  print(f"Total Y position: {pos_tot_y}")
  print(f"Step X position: {pos_step_x}")
  print(f"Step Y position: {pos_step_y}")
  np.savetxt("heatmap_data.csv", position, delimiter=",")

  np.savetxt("pos_data.csv", position, delimiter=",")

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  sc = ax.scatter(position[:, 0], position[:, 1], position[:, 2], c=timestamp, cmap='rainbow', marker='o')
  plt.colorbar(sc, label='Timestamp')
  ax.set_xlabel('X Position (m)')
  ax.set_ylabel('Y Position (m)')
  ax.set_zlabel('Z Position (m)')
  ax.set_title('Position over time')
  plt.savefig('data/xyz_plot.png')

  # XY Plot
  plt.figure()
  plt.scatter(position[:, 0], position[:, 1], c=timestamp, cmap='rainbow')
  plt.xlabel('X Position (m)')
  plt.ylabel('Y Position (m)')
  plt.title('XY Position Plot')
  plt.grid(True)
  plt.savefig('data/xy_plot.png')

  # # XZ Plot
  # plt.figure()
  # plt.scatter(position[:, 0], position[:, 2], c=timestamp, cmap='rainbow')
  # plt.xlabel('X Position (m)')
  # plt.ylabel('Z Position (m)')
  # plt.title('XZ Position Plot')
  # plt.grid(True)
  # plt.savefig('data/xz_plot.png')

  # # YZ Plot
  # plt.figure()
  # plt.scatter(position[:, 1], position[:, 2], c=timestamp, cmap='rainbow')
  # plt.xlabel('Y Position (m)')
  # plt.ylabel('Z Position (m)')
  # plt.title('YZ Position Plot')
  # plt.grid(True)
  # plt.savefig('data/yz_plot.png')

if True:
  main()
  plt.show()
