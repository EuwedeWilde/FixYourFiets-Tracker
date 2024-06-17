import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

df_raw_data = pd.read_csv('data/cal_data.csv')

def gyro_calibration():
  raw_gyro_data = np.array(df_raw_data[df_raw_data['sensor'] == 'gyro'][['gyro_x', 'gyro_y', 'gyro_z']].values)
  gyro_b = raw_gyro_data.mean(axis=0)
  print(f"gyro_b = np.array([{gyro_b[0]},{gyro_b[1]},{gyro_b[2]}])")

def acc_calibration():
  offsets = {}
  scale_factors = {}

  offsets['X'] = (df_raw_data.loc[df_raw_data['dir'] == 'x+', 'acc_x'].mean() + df_raw_data.loc[df_raw_data['dir'] == 'x-', 'acc_x'].mean()) / 2
  offsets['Y'] = (df_raw_data.loc[df_raw_data['dir'] == 'y+', 'acc_y'].mean() + df_raw_data.loc[df_raw_data['dir'] == 'y-', 'acc_y'].mean()) / 2
  offsets['Z'] = (df_raw_data.loc[df_raw_data['dir'] == 'z+', 'acc_z'].mean() + df_raw_data.loc[df_raw_data['dir'] == 'z-', 'acc_z'].mean()) / 2

  # Calculate scale factors for each axis
  scale_factors['X'] = 2 / (df_raw_data.loc[df_raw_data['dir'] == 'x+', 'acc_x'].mean() - df_raw_data.loc[df_raw_data['dir'] == 'x-', 'acc_x'].mean())
  scale_factors['Y'] = 2 / (df_raw_data.loc[df_raw_data['dir'] == 'y+', 'acc_y'].mean() - df_raw_data.loc[df_raw_data['dir'] == 'y-', 'acc_y'].mean())
  scale_factors['Z'] = 2 / (df_raw_data.loc[df_raw_data['dir'] == 'z+', 'acc_z'].mean() - df_raw_data.loc[df_raw_data['dir'] == 'z-', 'acc_z'].mean())

  print(f"acc_b = np.array([{offsets['X']},{offsets['Y']},{offsets['Z']}])")
  print(f"acc_A_1 = np.array([[{scale_factors['X']},0,0],\n\t\t    [0,{scale_factors['Y']},0],\n\t\t    [0,0,{scale_factors['Z']}]])")

class mag_calibration(object):
  MField = 1000
  def __init__(self, F=MField): 
    self.F   = F
    self.b   = np.zeros([3, 1])
    self.A_1 = np.eye(3)
      
  def run(self):
    data = np.array(df_raw_data[df_raw_data['sensor'] == 'mag'][['mag_x', 'mag_y', 'mag_z']].values)

    
    s = np.array(data).T
    M, n, d = self.__ellipsoid_fit(s)

    M_1 = linalg.inv(M)
    self.b = -np.dot(M_1, n)
    self.A_1 = np.real(self.F / np.sqrt(np.dot(n.T, np.dot(M_1, n)) - d) * linalg.sqrtm(M))

    result = [] 
    for row in data: 
      xm_off  = row[0]-self.b[0]
      ym_off  = row[1]-self.b[1]
      zm_off  = row[2]-self.b[2]
      xm_cal = xm_off *  self.A_1[0,0] + ym_off *  self.A_1[0,1]  + zm_off *  self.A_1[0,2] 
      ym_cal = xm_off *  self.A_1[1,0] + ym_off *  self.A_1[1,1]  + zm_off *  self.A_1[1,2] 
      zm_cal = xm_off *  self.A_1[2,0] + ym_off *  self.A_1[2,1]  + zm_off *  self.A_1[2,2] 

      result = np.append(result, np.array([xm_cal, ym_cal, zm_cal]))

    print(f"mag_b = np.array([{self.b[0][0]},{self.b[1][0]},{self.b[2][0]}])")
    print(f"mag_A_1 = np.array([[{self.A_1[0,0]},{self.A_1[0,1]},{self.A_1[0,2]}],\n\t\t    [{self.A_1[1,0]},{self.A_1[1,1]},{self.A_1[1,2]}],\n\t\t    [{self.A_1[2,0]},{self.A_1[2,1]},{self.A_1[2,2]}]])")


    result = result.reshape(-1, 3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    ax.scatter(data[:,0], data[:,1], data[:,2], marker='o', color='r')
    ax.scatter(result[:,0], result[:,1], result[:,2], marker='o', color='g')
    plt.show()


  def __ellipsoid_fit(self, s):
    D = np.array([s[0]**2., s[1]**2., s[2]**2.,
                  2.*s[1]*s[2], 2.*s[0]*s[2], 2.*s[0]*s[1],
                  2.*s[0], 2.*s[1], 2.*s[2], np.ones_like(s[0])])

    S = np.dot(D, D.T)
    S_11 = S[:6,:6]
    S_12 = S[:6,6:]
    S_21 = S[6:,:6]
    S_22 = S[6:,6:]

    C = np.array([[-1,  1,  1,  0,  0,  0],
                  [ 1, -1,  1,  0,  0,  0],
                  [ 1,  1, -1,  0,  0,  0],
                  [ 0,  0,  0, -4,  0,  0],
                  [ 0,  0,  0,  0, -4,  0],
                  [ 0,  0,  0,  0,  0, -4]])

    E = np.dot(linalg.inv(C), S_11 - np.dot(S_12, np.dot(linalg.inv(S_22), S_21)))

    E_w, E_v = np.linalg.eig(E)

    v_1 = E_v[:, np.argmax(E_w)]
    if v_1[0] < 0: v_1 = -v_1
    v_2 = np.dot(np.dot(-np.linalg.inv(S_22), S_21), v_1)

    M = np.array([[v_1[0], v_1[5], v_1[4]],
                  [v_1[5], v_1[1], v_1[3]],
                  [v_1[4], v_1[3], v_1[2]]])
    n = np.array([[v_2[0]],
                  [v_2[1]],
                  [v_2[2]]])
    d = v_2[3]

    return M, n, d

if __name__ == '__main__':
  gyro_calibration()
  acc_calibration()
  mag_calibration().run()
  



