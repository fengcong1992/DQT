import numpy as np
import pandas as pd
import math
import scipy.io as sio
import os

def sin_cos_encoding(features):
    ''' 
    datetime encoding
    
    Parameters
    ----------
    features : numpy array, date of month, date of week, hour, ...
        n_features*n_samples

    Returns
    -------
    enc_ft : numpy array, encoded time info
        n_samples*n_features

    '''
    n = features.shape[0]
    enc_ft = []
    for i in range(n):
        column = features[i,:]
        val = (2*math.pi*column/max(column))
        sin_values = [math.sin(x) for x in list(val)]
        cos_values = [math.cos(x) for x in list(val)] 
        enc_ft.append(sin_values)
        enc_ft.append(cos_values)
    
    enc_ft = np.swapaxes(np.asarray(enc_ft),0,1)
    return enc_ft

def normalization(test_data,train_data, method="method1", mask=False): 
    ''' 
    Parameters
    ----------
    test_data : numpy array
        n_samples*n_features
    train_data :  numpy array
        n_samples*n_features
    method: str
        "method1": (array -mean)/std
        "method2": (array -min)/(max-min)
    mask: bool
        True: Normalize non-Zero values of the array.
        False: Normalize whole arrays
    Returns
    -------
    test_data : numpy array
        n_samples*n_features
    train_data : numpy array
        n_samples*n_features 
    "method1": mean and std
    "method2": min and max
    '''
    if mask==False:
        if method=="method1": 
            train_mean = np.mean(train_data,axis = 0)
            train_std = np.std(train_data,axis = 0)
            test_data = (test_data-train_mean)/train_std
            train_data = (train_data-train_mean)/train_std
            return test_data, train_data, train_mean, train_std
        if method=="method2": 
            train_min = np.min(train_data,axis = 0)
            train_max = np.max(train_data,axis = 0)
            test_data = (test_data-train_min)/(train_max - train_min)
            train_data = (train_data-train_min)/(train_max - train_min)
            return test_data, train_data, train_min, train_max
    else:
        n_features = train_data.shape[1] 
        if method=="method1":
            train_mean = []
            train_std = []
            train =[]
            test = []
            for i in range(n_features):
                train_fts = train_data[:,i] 
                test_fts = test_data[:,i] 
                i_mean = np.mean(train_fts[train_fts!=0])
                i_std = np.std(train_fts[train_fts!=0])
                a = np.where(test_fts!=0,(test_fts[test_fts!=0]-i_mean)/i_std)
                b = np.where(train_fts!=0,(train_fts[train_fts!=0]-i_mean)/i_std)
                train.append(b)
                test.append(a) 
                train_mean.append(i_mean)
                train_std.append(i_std)
            return np.transpose(np.array(test)), np.transpose(np.array(train)), train_mean, train_std
        if method=="method2":
            train_min = []
            train_max = []
            train = []
            test=[]
            for i in range(n_features):
                train_fts = train_data[:,i] 
                test_fts = test_data[:,i]   
                i_min = np.min(train_fts[train_fts!=0])
                i_max = np.max(train_fts[train_fts!=0])  
                a = np.where(test_fts!=0,(test_fts-i_min)/(i_max-i_min), test_fts)
                b = np.where(train_fts!=0,(train_fts-i_min)/(i_max-i_min),train_fts)  
                train.append(b)
                test.append(a)
                train_min.append(i_min)
                train_max.append(i_max)
            return np.transpose(np.array(test)), np.transpose(np.array(train)), train_min, train_max
    

def CheckValidity(data,timeStamps,zenith, startT, endT, ndays):
    """ 
    Parameters
    ----------
    data : 
        X or y arrays to be transformed.
    timeStamps : 
        timestamps array.
    zenith : 
        daytime index array corresponding to timestamps.
    startT : pd.Timestamp
        start time of the sample
    endT : pd.Timestamp
        ending time of the sample.
    ndays : int
        duration of the sample.

    Returns
    -------
    data: numpy array
        a sample.
    timestamps: pd.series
        timestamps corresponding to the data sample
    zenith: numpy array
        daytime index corresponding to the data sample.

    """
    t1 = timeStamps[timeStamps==startT].index
    t2 = timeStamps[timeStamps==endT].index
    if len(t1)==0 or len(t2)==0 or (t2[0]-t1[0]+1)!=ndays*24:
        print('starting time not selected:',startT)
        return None,None,None
    else:
        return data[t1[0]:t2[0]+1,:],timeStamps[t1[0]:t2[0]+1],zenith[t1[0]:t2[0]+1]
        
    
def CreateSamples(X, y, timeStamps, zenith, startT, endT, hist_days, pred_days, lead_time, stride=1):
    """
    Parameters
    ----------
    X : numpy array
        number of samples* number of features.
    y : numpy array
        number of samples*1.
    timeStamps: pd.Series of timestamps of X and y
    startT: pd.Timestamp, datetime of the starting hour of the first current day
    endT: pd.Timestamp, datetime of the last hour of last prediction day
    hist_days : list of negative values
        days prior to current day as predictors.
    pred_days : int
        number of forecasting days, represents number of days
    lead_time: int
        number of lead days
    stride : int
        stride of rolling window, represents number of days.

    Returns
    -------
    newX : numpy array
        number of samples * number of timesteps * number of features.
    newy : numpy array
        number of samples * number of timesteps * number of features..
    pred_time: list of pd.series
        timestamps of prediction values.
    y_zenith: numpy array
        number of samples *forecasting horizon. 0: night hour. 1: daytime hour.
    """
    n = len(hist_days)
    newX = []
    newy = [] 
    pred_time = []
    pred_zenith = []
    predStart = startT+pd.Timedelta(lead_time,unit='d')
    predEd = predStart+pd.Timedelta(hours=23)
    while predEd<=endT:
        # check validity of prediction days 
        pred,pred_time_tmp, pred_zenith_tmp = CheckValidity(y,timeStamps,zenith,predStart,
                                                        predEd,pred_days)  
        if pred is not None:
            pred = np.reshape(pred,(-1,pred_days*24)) 
            # check validity of historical days
            hist = []
            for i in range(len(hist_days)): 
                tmp,_,_ = CheckValidity(X, timeStamps, zenith, startT+pd.Timedelta(days=hist_days[i]), 
                                     startT+pd.Timedelta(days=hist_days[i])+pd.Timedelta(hours=23),
                                     ndays=1)
                if tmp is not None:
                    tmp = np.reshape(tmp,(24,X.shape[1]))
                    hist.append(tmp)
                else: 
                    break
            if len(hist)==n:
                hist = np.reshape(np.array(hist),(1,-1,X.shape[1]))
                newX.append(hist)
                newy.append(pred) 
                pred_time.append(pred_time_tmp)
                pred_zenith.append(pred_zenith_tmp)
        startT += pd.Timedelta(days=stride)
        predStart = startT+pd.Timedelta(lead_time,unit='d')
        predEd = predStart+pd.Timedelta(hours=23)
        # print('startT:', predStart)
        # print('endT:', predEd)
        
    newX = np.reshape(np.array(newX),(-1,n*24,X.shape[1]))
    # pred_time = np.reshape(np.array(pred_time),(-1,pred_days*24))
    newy = np.reshape(np.array(newy),(-1,pred_days*24))
    return newX, newy,pred_time,np.array(pred_zenith)

if __name__=="__main__": 
# =============================================================================
#     Load datasets
# ============================================================================= 
    data1 = pd.read_excel('data/HECOData_Clean_2017_with_zenith.xlsx')
    data2 = pd.read_excel('data/HECOData_Clean_2018_with_zenith.xlsx')
    weatherData = pd.read_csv('data/HI_NSRDB.csv')
    # persistence = pd.read_csv('data/normalized_persistence_results_v2.csv')
    # persistence['TimeStamp'] = persistence['TimeStamp'].astype('datetime64') 
    ### merge datasets
    data1['TimeStamp'] = pd.to_datetime(data1['TimeStamp'],format='%m/%d/%Y %H:%M')
    data2['TimeStamp'] = pd.to_datetime(data2['TimeStamp'],format='%m/%d/%Y %H:%M')
    weatherData = weatherData.rename(columns={'time_index':'TimeStamp'})
    # s2 = pd.to_datetime(weatherData['TimeStamp'][17520:],format="b'%Y-%m-%d %H:%M:%S'")
    weatherData.loc[0:17519,('TimeStamp')] = pd.to_datetime(weatherData.loc[0:17519,('TimeStamp')],format="b'%Y-%m-%dT%H:%M:%S.000000000'")
    weatherData.loc[17520:,('TimeStamp')] = pd.to_datetime(weatherData.loc[17520:,('TimeStamp')],format="b'%Y-%m-%d %H:%M:%S'")
    weatherData['TimeStamp'] = weatherData['TimeStamp'].astype('datetime64') 
    weatherData.loc[17520:,('air_temperature','surface_pressure')] = \
                weatherData.loc[17520:,('air_temperature','surface_pressure')]/10
    weatherData.loc[0:17519,('wind_direction')] = \
                weatherData.loc[0:17519,('wind_direction')]/10
    # data = pd.merge(persistence, weatherData, on ='TimeStamp')
    # data.reset_index(drop=True)
    trainData = pd.merge(data1,weatherData,on ='TimeStamp')
    testData = pd.merge(data2,weatherData,on = 'TimeStamp')
    
    # trainData = data[data['TimeStamp']<pd.Timestamp(year=2018,month=1,day=1,hour=11)]
    # testData = data[data['TimeStamp']>=pd.Timestamp(year=2018,month=1,day=1,hour=11)]
    # trainData = pd.merge(trainData,data1[['Zenith', 'TimeStamp']],on = "TimeStamp")
    trainData.reset_index(drop=True)
    # testData = pd.merge(testData, data2[['Zenith', 'TimeStamp']], on = "TimeStamp")
    testData.reset_index(drop=True)
    # trainData.reset_index(drop=True)
    # testData = pd.merge(data2, weatherData, on ='TimeStamp')
    # testData.reset_index(drop=True)
# =============================================================================
#     Data processing
# =============================================================================
    # ### dictionary {'city': n}, where n represents UTC-n
    timeZone = {'Amity':7, 'Donalsonville':4, 'SanAntonio':5,'Waianae':10}
    
    # normalization: (values-mean)/std
    # var1 = ['Load','air_temperature','relative_humidity','surface_pressure','wind_direction','wind_speed']
    NSRDB_var1 = ['air_temperature','relative_humidity','surface_pressure','wind_direction','wind_speed']
    # HECO_var1 = ['Load']
    train1= trainData[NSRDB_var1]
    test1 = testData[NSRDB_var1]
    test1, train1,mean1,s1 = normalization(test1.to_numpy(), train1.to_numpy(),method='method1',mask=False)
   
    # masked normalization of PV: (PV-min)/(max-min); keepping 0.
    # var2 = ['PV','ghi','dni','dhi','clearsky_ghi','clearsky_dni','clearsky_dhi']
    NSRDB_var2 = ['ghi','dni','dhi','clearsky_ghi','clearsky_dni','clearsky_dhi']
    # HECO_var2 = ['PV']
    train2 = trainData[NSRDB_var2]
    test2 = testData[NSRDB_var2]
    test2, train2, min1,max1 = normalization(test2.to_numpy(),train2.to_numpy(),method='method2',mask=True)
     
    # encoding time information with sin cos functions
    time_train = trainData['TimeStamp']
    time_test = testData['TimeStamp']
    enc_train = np.vstack((time_train.dt.month.to_numpy(),time_train.dt.day.to_numpy(),\
                                time_train.dt.weekday.to_numpy(),time_train.dt.hour.to_numpy()))
    enc_test = np.vstack((time_test.dt.month.to_numpy(),time_test.dt.day.to_numpy(),\
                               time_test.dt.weekday.to_numpy(),time_test.dt.hour.to_numpy()))
    enc_train = sin_cos_encoding(enc_train)
    enc_test = sin_cos_encoding(enc_test)
    
    # # check time intervals 
    # datetime1 = time_train[1:].reset_index(drop=True) 
    # datetime2 = time_test[1:].reset_index(drop=True)
    # diff1 = (datetime1-time_train[:-1])/ pd.Timedelta(hours=1)
    # diff2 = (datetime2-time_test[:-1])/ pd.Timedelta(hours=1)
    # idx1 = np.where(diff1!=1)
    # idx2 = np.where(diff2!=1)
    # breakpts1 = []
    # breakpts2 = []
    # for i in range(len(idx1)):
    #     breakpts1.append(idx1[i])
    # for i in range(len(idx2)):
    #     breakpts2.append(idx1[i]) 
    # print(diff1.value_counts(dropna=False))
    # print(diff2.value_counts(dropna=False))
     
    # trainy = np.reshape(trainData['targets'].to_numpy(),(-1,1))
    # testy = np.reshape(testData['targets'].to_numpy(),(-1,1))
    trainy = trainData['NL'].to_numpy()
    testy = testData['NL'].to_numpy()
    peak = max(trainy)
    trainy = trainy/peak
    testy = testy/peak
    
    zenith_train = np.where(trainData['Zenith'].to_numpy()>87,0,1)
    zenith_test = np.where(testData['Zenith'].to_numpy()>87,0,1)
    
    ### factors: NL, PV, Load, weather and time
    # trainX = np.concatenate((enc_train,train1,train2,trainy),axis = 1) 
    # testX = np.concatenate((enc_test,test1,test2,testy),axis = 1) 
    ### factors: only weather and time
    trainX = np.concatenate((enc_train,train1,train2),axis = 1) 
    testX = np.concatenate((enc_test,test1,test2),axis = 1) 
    ### factors: only NL and Time
    # trainX = np.concatenate((enc_train,trainy),axis = 1) 
    # testX = np.concatenate((enc_test,testy),axis = 1) 
    ### factors: HECO d-1, NSRDB d1
    # train_NSRDB = np.concatenate((enc_train,train1[:,1:],train2[:,1:]),axis = 1) 
    # train_HECO = np.concatenate((train1[:,0:1],train2[:,0:1],trainy),axis = 1) 
    # test_NSRDB = np.concatenate((enc_test,test1[:,1:],test2[:,1:]),axis = 1)  
    # test_HECO = np.concatenate((test1[:,0:1],test2[:,0:1],testy),axis = 1)  
    ### factors: persistence and NSRDB
    # trainX = np.concatenate((enc_train,train1,train2,trainData.iloc[:,1:12].to_numpy()),axis = 1)
    # testX = np.concatenate((enc_test,test1,test2,testData.iloc[:,1:12].to_numpy()),axis = 1)
    ### save stats
    # method1_variables = ['Load','air_temperature','relative_humidity','surface_pressure',\
    #                 'wind_direction','wind_speed']
    # method2_variables = ['PV','ghi','dni','dhi','clearsky_ghi','clearsky_dni','clearsky_dhi']
    # with open('data/HECO_and_NSRDB_stats.txt','w') as f: 
    #     for i in range(len(method1_variables)):
    #         f.write(f'{method1_variables[i]} mean and std: {mean1[i]}, {s1[i]}\n')
    #     for i in range(len(method2_variables)):
    #         f.write(f'{method2_variables[i]} min and max: {min1[i]}, {max1[i]}\n')
    #     f.write('NL train peak:\t'+str(peak[0])+'\n')
# # =============================================================================
# #     creating samples
# # ============================================================================= 
    foldername = 'NSRDB_lag0_lt0_pred1d_daytime_only'
    path = os.getcwd()+'/data/'+foldername
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'folder {path} is created!')
    
# =============================================================================
#     # heco and nsrdb
# =============================================================================
    # train_HECO = np.reshape(train_HECO,(-1,24,train_HECO.shape[1]))
    # test_HECO = np.reshape(test_HECO,(-1,24,test_HECO.shape[1]))
    # train_NSRDB = np.reshape(train_NSRDB,(-1,24,train_NSRDB.shape[1]))
    # test_NSRDB = np.reshape(test_NSRDB,(-1,24,test_NSRDB.shape[1]))
    
    # ### missing one day between index 193-194
    # trainX1 = np.concatenate((train_HECO[0:192,:,:],train_NSRDB[2:194,:,:]),axis =2)
    # trainX2 = np.concatenate((train_HECO[194:-2,:,:],train_NSRDB[196:,:,:]),axis =2)
    # trainX = np.concatenate((trainX1,trainX2),axis = 0)
    # trainy = np.reshape(trainy,(-1,24))
    # trainy = np.concatenate((trainy[2:194,:],trainy[196:,:]),axis=0)
    # zenith_train = np.reshape(zenith_train,(-1,24))
    # zenith_train = np.concatenate((zenith_train[2:194,:],zenith_train[196:,:]),axis=0)
    # time_train = np.reshape(time_train.to_numpy(),(-1,24))
    # time_train = np.concatenate((time_train[2:194,:],time_train[196:,:]),axis=0)
    
    # testX = np.concatenate((test_HECO[0:-2,:,:],test_NSRDB[2:,:,:]),axis =2)
    # testy = np.reshape(testy,(-1,24))
    # testy = testy[2:,:]
    # zenith_test = np.reshape(zenith_test,(-1,24))
    # zenith_test = zenith_test[2:,:]
    # time_test = np.reshape(time_test.to_numpy(),(-1,24))
    # time_test = time_test[2:,:]
    
    # np.savez(path+'/2017_'+foldername+'_TRAIN.npz',X=trainX,y=trainy,daytimeIdx = zenith_train,
    #           targetsTime = time_train)
    # np.savez(path+'/2018_'+foldername+'_TEST.npz',X=testX,y=testy,daytimeIdx = zenith_test,
    #           targetsTime = time_test)
    
# =============================================================================
#     historical timestamps to predict future timestamps
# =============================================================================
    # startT = pd.Timestamp(year=2017,month=2,day=1,hour=11)
    # endT= pd.Timestamp(year=2018,month=1,day=1,hour=10)
    
    # X, y, y_time,y_zenith = CreateSamples(trainX, trainy, time_train, zenith_train,startT, 
    #                             endT, hist_days=[0],pred_days=1,lead_time=0,
    #                             stride=1)
    # np.savez(path+'/2017_'+foldername+'_TRAIN.npz',X=X,y=y,daytimeIdx = y_zenith,
    #           targetsTime = y_time)
    
    # startT = pd.Timestamp(year=2018,month=1,day=1,hour=11)
    # endT= pd.Timestamp(year=2018,month=9,day=30,hour=10)
    
    # X, y, y_time,y_zenith = CreateSamples(testX,testy,time_test, zenith_test,startT, 
    #                             endT, hist_days=[0],pred_days=1,lead_time=0,
    #                             stride=1)
    # np.savez(path+'/2018_'+foldername+'_TEST.npz',X=X,y=y,daytimeIdx = y_zenith,
    #           targetsTime = y_time)
    
# =============================================================================
#     # only forecasting
# =============================================================================
    # trainX = np.reshape(trainX,(-1,24,trainX.shape[1]))
    # trainy = np.reshape(trainy,(-1,24))
    # zenith_train = np.reshape(zenith_train,(-1,24))
    # time_train = np.reshape(time_train.to_numpy(),(-1,24))
    # np.savez(path+'/2017_'+foldername+'_TRAIN.npz',X=trainX,y=trainy,daytimeIdx = zenith_train,
    #           targetsTime = time_train)   
    # testX = np.reshape(testX,(-1,24,testX.shape[1]))
    # testy = np.reshape(testy,(-1,24))
    # zenith_test = np.reshape(zenith_test,(-1,24))
    # time_test = np.reshape(time_test.to_numpy(),(-1,24))
    # np.savez(path+'/2018_'+foldername+'_TEST.npz',X=testX,y=testy,daytimeIdx = zenith_test,
    #               targetsTime = time_test)
    
# =============================================================================
#     daytime only
# =============================================================================
    # set daytime range as 8AM-20PM
    startT = pd.Timestamp(year = 2017, month=1,day=1,hour=11)
    endT = pd.Timestamp(year=2018, month=1,day=1,hour=10)
    trainy = np.reshape(trainy,(-1,1))
    X, y, y_time,y_zenith = CreateSamples(trainX, trainy, time_train, zenith_train,startT, 
                                endT, hist_days=[0],pred_days=1,lead_time=0,
                                stride=1)
    # from UTC 18:00PM - (day+1) 10AM
    X = X[:,6:19,:]
    y = y[:,6:19]
    y_time = np.array([i[6:19].reset_index(drop=True) for i in y_time])
    y_zenith = y_zenith[:,6:19]
    
    np.savez(path+'/2017_'+foldername+'_TRAIN.npz',X=X,y=y,daytimeIdx = y_zenith,
              targetsTime = y_time)
    
    startT = pd.Timestamp(year=2018,month=1,day=1,hour=11)
    endT = pd.Timestamp(year=2018, month = 9,day=30,hour=10)
    testy = np.reshape(testy,(-1,1))
    X, y, y_time, y_zenith = CreateSamples(testX, testy, time_test, zenith_test, startT, 
                                           endT, hist_days=[0], pred_days=1, lead_time=0)
    X = X[:,6:19,:]
    y = y[:,6:19]
    y_time = np.array([i[6:19].reset_index(drop=True) for i in y_time])
    y_zenith = y_zenith[:,6:19]
    np.savez(path+'/2018_'+foldername+'_TEST.npz',X=X,y=y,daytimeIdx = y_zenith,
              targetsTime = y_time)