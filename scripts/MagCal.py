import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt
import pandas as pd

 #https://github.com/nliaudat/magnetometer_calibration/blob/main/calibrate.py
 #corrected code S. James Remington
 
class Magnetometer(object):
  MField = 1000
  def __init__(self, F=MField): 
    self.F   = F
    self.b   = np.zeros([3, 1])
    self.A_1 = np.eye(3)
      
  def run(self):
    df_raw_data = pd.read_csv('data/mag_cal_data.csv')
    data = np.array(df_raw_data[['mag_x', 'mag_y', 'mag_z']])
    print("shape of data:",data.shape)
    print("First 5 rows raw:\n", data[:5])
    
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
        
        
        
if __name__=='__main__':
  Magnetometer().run()