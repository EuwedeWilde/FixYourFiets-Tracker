import scripts.dataRead as dr
import scripts.devicePosition as dp

while True:
  data = dr.readSerial()
  dp.dataProcessing(data[1], data[0])
  
