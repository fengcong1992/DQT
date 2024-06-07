import numpy as np
import os

fn = ["Amity_competition_lag0d_lt0d_pred1d","Donalsonville_competition_lag0d_lt0d_pred1d",
      "San_Antonio_competition_lag0d_lt0d_pred1d","Waianae_competition_lag0d_lt0d_pred1d"]
start_id = [5,6,6,6]
end_id = [20,20,20,19]

for i in range(4):
    path = os.getcwd()+'/data/'+fn[i]+'_daytime'
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'folder {path} is created!')
    testData = np.load("data/"+fn[i]+"/"+fn[i]+"_TEST.npz", allow_pickle=True)
    trainData = np.load("data/"+fn[i]+"/"+fn[i]+"_TRAIN.npz", allow_pickle=True)
    X = testData['X'][:,start_id[i]:end_id[i],:]
    y = testData['y'][:,start_id[i]:end_id[i]]
    time = testData['targetsTime'][:,start_id[i]:end_id[i]]
    dt = testData['daytimeIdx'][:,start_id[i]:end_id[i]]
    np.savez(f"{path}/{fn[i]}_daytime_TEST.npz",X=X,y=y,daytimeIdx=dt,targetsTime=time)
    
    X = trainData['X'][:,start_id[i]:end_id[i],:]
    y = trainData['y'][:,start_id[i]:end_id[i]]
    time = trainData['targetsTime'][:,start_id[i]:end_id[i]]
    dt = trainData['daytimeIdx'][:,start_id[i]:end_id[i]]
    np.savez(f"{path}/{fn[i]}_daytime_TRAIN.npz",X=X,y=y,daytimeIdx=dt,targetsTime=time)