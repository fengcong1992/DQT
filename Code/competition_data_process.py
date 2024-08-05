import numpy as np
import pandas as pd
import math
import scipy.io as sio
import os
from pysolar.solar import *

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

    ### prediction starts from 1 AM in local time
    time_zone = {'city':['Amity','Donalsonville','San_Antonio','Waianae'],
                 'start_time':[8,5,6,11],'latitude':[45.114559,31.044241,29.424122,21.446911],
                           'longtitude':[-123.204903,-84.879128,-98.493629,-158.188736]}
    
    files = os.listdir(os.getcwd()+'/data/')
    NL_files = [fn for fn in files if 'Net_Load' in fn]
    weather_files = [fn for fn in files if 'historical_data' in fn]
    
    for i in range(4): 
        # =============================================================================
        #     Load datasets
        # =============================================================================
        ### read and combine two years' net load csv files
        iNL = [fn for fn in NL_files if time_zone['city'][i] in fn] 
        assert len(iNL)==2, f"no exact 2 Net load files of {time_zone['city'][i]}."
        NLData1 = pd.read_csv('data/'+iNL[0],skiprows=[0,1])
        NLData2 = pd.read_csv('data/'+iNL[1],skiprows=[0,1])
        NLData = pd.concat([NLData1,NLData2],ignore_index=True)
        NLData = NLData.rename(columns={'timestamp':'TimeStamp','value':'NL'})
        NLData['TimeStamp'] = pd.to_datetime(NLData['TimeStamp'],utc=True)
        NLData = NLData.sort_values(by='TimeStamp')
        NLData = NLData.drop(['quality_flag'],axis=1)
        ### read the weather file
        weather_fn = [fn for fn in weather_files if time_zone['city'][i] in fn]
        assert len(weather_fn)==1, f"no exact 1 weather files of {time_zone['city'][i]}."
        wData = pd.read_csv('data/'+weather_fn[0])
        wData['TimeStamp'] = pd.to_datetime(wData['TimeStamp'],utc=True)
        ### combine weather and net load files
        data = pd.merge(NLData,wData,how ='inner', on='TimeStamp')
        data = data.reset_index(drop=True) 
        
        ### check time intervals
        tmp_time1 = data.iloc[1:,0].reset_index(drop=True) 
        tmp_time2 = data.iloc[:-1,0].reset_index(drop=True) 
        diff1 = (tmp_time1-tmp_time2)/ pd.Timedelta(hours=1)
        print(f"count time differences of {time_zone['city'][i]}",diff1.value_counts(dropna=False))   
        
        # =============================================================================
        #         split training and testing datasets
        # =============================================================================
        trainData = data[data['TimeStamp']<pd.Timestamp(year=2023,month=5,day=1,
                                                        hour=time_zone['start_time'][i],tz='UTC')].reset_index(drop=True)
        testData = data[data['TimeStamp']>=pd.Timestamp(year=2023,month=5,day=1,
                                                        hour=time_zone['start_time'][i],tz='UTC')].reset_index(drop=True)
        train_fts = trainData.drop(['TimeStamp','NL'],axis=1).to_numpy()
        test_fts = testData.drop(['TimeStamp','NL'],axis=1).to_numpy()
        
        # =============================================================================
        #         # normalize features
        # ============================================================================= 
        test_fts, train_fts, train_mean, train_std = normalization(test_fts,train_fts, method="method1", mask=False)
        ### encoding time information
        time_train = trainData['TimeStamp']
        time_test = testData['TimeStamp']
        enc_train = np.vstack((time_train.dt.month.to_numpy(),time_train.dt.day.to_numpy(),\
                                    time_train.dt.weekday.to_numpy(),time_train.dt.hour.to_numpy()))
        enc_test = np.vstack((time_test.dt.month.to_numpy(),time_test.dt.day.to_numpy(),\
                                   time_test.dt.weekday.to_numpy(),time_test.dt.hour.to_numpy()))
        enc_train = sin_cos_encoding(enc_train)
        enc_test = sin_cos_encoding(enc_test) 
        
        ### save stats of normalization
        with open(f"data/{time_zone['city'][i]}_competition_data_features_stats.txt",'w') as f:
            for j in range(len(testData.columns)-2):
                f.write(f'{testData.columns[j+2]}\n mean: {train_mean[j]:.4f}, std: {train_std[j]:.4f}\n') 
        trainX = np.concatenate((enc_train,train_fts),axis=1)
        testX = np.concatenate((enc_test,test_fts),axis=1)
        trainy = np.reshape(trainData['NL'].to_numpy(),(-1,1))
        testy = np.reshape(testData['NL'].to_numpy(),(-1,1))
        
        # =============================================================================
        #         # generating Zenith
        # =============================================================================
        zenith_train = []
        zenith_test = []
        # time_train = time_train.to_numpy()
        # time_test = time_test.to_numpy() 
        for j in range(len(time_train)):
            idt = time_train[j].to_pydatetime() 
            al = get_altitude(time_zone['latitude'][i], time_zone['longtitude'][i], idt)
            zenith_train.append(90-al)
        for j in range(len(time_test)):
            idt = time_test[j].to_pydatetime()
            al = get_altitude(time_zone['latitude'][i], time_zone['longtitude'][i], idt)
            zenith_test.append(90-al)
        zenith_train = np.where(np.array(zenith_train)>87,0,1)
        zenith_test = np.where(np.array(zenith_test)>87,0,1)
        # =============================================================================
        #         # create samples
        # =============================================================================
        foldername = f"{time_zone['city'][i]}_competition_lag0d_lt0d_pred1d"
        path = os.getcwd()+'/data/'+foldername
        if not os.path.exists(path):
            os.makedirs(path)
            print(f'folder {path} is created!')
        startT = pd.Timestamp(year=2022,month=1,day=1,hour=time_zone['start_time'][i],tz='UTC')
        endT= pd.Timestamp(year=2023,month=5,day=1,hour=time_zone['start_time'][i]-1,tz='UTC')
        X, y, y_time,y_zenith = CreateSamples(trainX, trainy, time_train, zenith_train,startT, 
                                    endT, hist_days=[0],pred_days=1,lead_time=0,stride=1)
        y_time = np.array(y_time)
        np.savez(path+'/'+foldername+'_TRAIN.npz',X=X,y=y,daytimeIdx = y_zenith,
                  targetsTime = y_time)
        
        startT = pd.Timestamp(year=2023,month=5,day=1,hour=time_zone['start_time'][i],tz='UTC')
        if i==2:
            endT= pd.Timestamp(year=2023,month=5,day=30,hour=time_zone['start_time'][i]-1,tz='UTC')
        else:
            endT= pd.Timestamp(year=2023,month=5,day=31,hour=time_zone['start_time'][i]-1,tz='UTC')
        X, y, y_time,y_zenith = CreateSamples(testX, testy, time_test, zenith_test, startT, 
                                    endT, hist_days=[0],pred_days=1,lead_time=0,stride=1)
        y_time = np.array(y_time)
        np.savez(path+'/'+foldername+'_TEST.npz',X=X,y=y,daytimeIdx = y_zenith,
                  targetsTime = y_time)