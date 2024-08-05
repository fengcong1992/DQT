# add weekday enc
import scipy.io as sio
import numpy as np
import math

# =============================================================================
#  datetime encoding
# =============================================================================
def sin_cos_encoding(features):
    n = features.shape[0]
    enc_ft = []
    for i in range(n):
        column = features[i,:]
        val = (2*math.pi*(column-min(column))/(max(column)-min(column)))
        sin_values = [math.sin(x) for x in list(val)]
        cos_values = [math.cos(x) for x in list(val)] 
        enc_ft.append(sin_values)
        enc_ft.append(cos_values)
    
    enc_ft = np.swapaxes(np.asarray(enc_ft),0,1)
    return enc_ft

filename = "combined_WaianaeHistoricalData"
data = sio.loadmat("data/"+filename+".mat")['data']
timestamp = sio.loadmat("data/combined_AmityHistoricalData.mat")['timstamp']

weekdays = [6,7,1,2,3,4,5]
weekdays = [[i]*24 for i in weekdays] 
n_Day = round(len(timestamp)/24/7)+1
weekdays = [j for i in weekdays for j in i]
weekdays = weekdays*n_Day
weekdays = weekdays[0:len(timestamp)]
weekdays = np.reshape(np.asarray(weekdays),(1,-1))
enc_weekdays = sin_cos_encoding(weekdays)
data = np.concatenate((enc_weekdays,data),axis = 1)

sio.savemat("data/"+filename+"_v2.mat",{"data":data,"timestamp":timestamp})