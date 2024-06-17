from flask import Flask, render_template, jsonify
import numpy as np
import pandas as pd
import imufusion as imu
from dataclasses import dataclass
from scipy.interpolate import interp1d

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/data')
def get_data():
  data = np.genfromtxt("data/track_data.csv", delimiter=",", skip_header=1)

  sample_rate = 400  
  timestamp = data[:, 9]
  gyroscope = data[:, 3:6]
  accelerometer = data[:, 0:3]
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

  is_moving = np.empty(len(timestamp))

  for index in range(len(timestamp)):
    is_moving[index] = np.sqrt(acceleration[index].dot(acceleration[index])) > 3  # threshold = 3 m/s/s

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

  velocity_drift = np.zeros((len(timestamp), 3))

  for is_moving_period in is_moving_periods:
    start_index = is_moving_period.start_index
    stop_index = is_moving_period.stop_index
    if start_index == -1 or stop_index == -1 or stop_index <= start_index:
      continue

    t = [timestamp[start_index], timestamp[stop_index]]
    x = [velocity[start_index, 0], velocity[stop_index, 0]]
    y = [velocity[start_index, 1], velocity[stop_index, 1]]
    z = [velocity[start_index, 2], velocity[stop_index, 2]]
    t_new = timestamp[start_index:(stop_index + 1)]

    # Debugging prints
    print(f'start_index: {start_index}, stop_index: {stop_index}')
    print(f't: {t}, x: {x}, y: {y}, z: {z}')
    print(f't_new length: {len(t_new)}, z length: {len(z)}')

    if len(t) != len(z):
      print(f'Skipping due to unequal lengths: t={t}, z={z}')
      continue

    velocity_drift[start_index:(stop_index + 1), 0] = interp1d(t, x)(t_new)
    velocity_drift[start_index:(stop_index + 1), 1] = interp1d(t, y)(t_new)
    velocity_drift[start_index:(stop_index + 1), 2] = interp1d(t, z)(t_new)

  velocity = velocity - velocity_drift

  position = np.zeros((len(timestamp), 3))

  for index in range(1, len(timestamp)):  # Start from 1 to avoid using position[-1]
    position[index] = position[index - 1] + delta_time[index] * velocity[index]

  return jsonify({
    'timestamp': timestamp.tolist(),
    'position': position.tolist(),
    'euler': euler.tolist()
  })

if __name__ == '__main__':
  app.run(debug=True)
