import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from models.measurements import measurements
import csv
import pandas as pd

def CRPSS(testData,persisDF,mdl_data,scale):
    ''' 
    ----------
    testData : npz file
        include testing data daytime index: "daytimeIdx", timestamps:"targetsTime" 
    persisDF : pd.DataFrame
        include timestamps, percentiles, and .
    mdl_data : npz file
        containing model predictions:"predictions" and model targets: "targets".
    scale: str
        "daily"--calculating daily crpss
        "hourly"--calculating hourly crpss
    Returns
    -------
    crpss : float or array
        average crpss value or daily or hourly crpss array.
    timeStamps: np array
        hourly timestamps of predictions
    '''
    daytimeIdx = testData['daytimeIdx']
    timeStamps = testData['targetsTime']
    mdl_pred = mdl_data['predictions']
    mdl_tar = mdl_data['targets']
    # print(f'timeStamps: {timeStamps}')
    # print()
    persisDF['TimeStamp'] = pd.to_datetime(persisDF['TimeStamp'])
    mdl_time = pd.DataFrame(pd.to_datetime(timeStamps.flatten()), columns=['TimeStamp'])
    persisDF = mdl_time.merge(persisDF, how='left', on='TimeStamp')
    
    if scale=='daily':
        persis_pred = np.reshape(persisDF.iloc[:,1:12].values,(-1,24,11))
        mdl_loss = measurements(mdl_pred, mdl_tar)
        persis_loss = measurements(persis_pred,mdl_tar)
        mdl_crps = mdl_loss.CRPS_array_dayloss(daytimeIdx)
        persis_crps = persis_loss.CRPS_array_dayloss(daytimeIdx)
        crpss = 1-np.array(mdl_crps)/np.array(persis_crps)
        return crpss,mdl_crps,persis_crps
    if scale=='hourly':
        persis_pred = persisDF.iloc[:,1:12].values
        daytimeIdx = daytimeIdx.flatten()
        mdl_pred = np.reshape(mdl_pred,(-1,11))
        mdl_tar = mdl_tar.flatten()  
        # print(persis_pred.shape)
        # print(mdl_tar.shape)
        mdl_loss = measurements(mdl_pred, mdl_tar)
        persis_loss = measurements(persis_pred,mdl_tar)
        mdl_crps = mdl_loss.CRPS_array_dayloss(daytimeIdx)
        persis_crps = persis_loss.CRPS_array_dayloss(daytimeIdx)
        crpss = 1-np.array(mdl_crps)/np.array(persis_crps)
        timeStamps = timeStamps.flatten()
        return crpss,mdl_crps,persis_crps,timeStamps[daytimeIdx==1]
  
def plotSingleVar(var,xlb,ylb,xsticks=[]):
    fig, ax = plt.subplots()
    if len(xsticks)==0:
        ax.plot(list(range(len(var))),var) 
    else:
        ax.plot(xsticks,var) 
    plt.xlabel(xlb)
    plt.ylabel(ylb)
    plt.show()

def collectLoss(expPath,test_Path,pred_folders): 
    ''' 
    Parameters
    ----------
    expPath : str
        the saving path of all prediction experiments.
    test_Path : list of str
        paths of testing data corresponding to predictions.
    pred_folders : list of str
        folder names of predictions.

    Returns
    -------
    rslt_dic : dictionary
        including experiment names and corresponding measurements.

    '''
    rslt_dic = {'experiment_name':[],'CRPS':[],'CRPS_sorted':[],'CRPS_dayloss':[],\
                'CRPS_dayloss_sorted':[],'pinball_loss':[],'pinball_loss_sorted':[]}
    n = len(pred_folders) 
    for i in range(n): 
        output = np.load(expPath+'/'+pred_folders[i]+'/predictions/best_predictions.npz')
        testData = np.load('data/'+test_Path[i]) 
        pred = output['predictions']  
        tar = output['targets']
        daytimeIdx = testData['daytimeIdx']
        loss = measurements(pred, tar, regweight=7)
        rslt_dic['experiment_name'].append(pred_folders[i]) 
        rslt_dic['CRPS'].append(loss.CRPS_loss())
        rslt_dic['pinball_loss'].append(loss.pinball_loss())
        rslt_dic['CRPS_dayloss'].append(loss.CRPS_dayloss(daytimeIdx)) 
        pred = np.sort(pred,axis=2)
        loss = measurements(pred, tar, regweight=5) 
        rslt_dic['CRPS_sorted'].append(loss.CRPS_loss()) 
        rslt_dic['pinball_loss_sorted'].append(loss.pinball_loss())
        rslt_dic['CRPS_dayloss_sorted'].append(loss.CRPS_dayloss(daytimeIdx)) 
    return rslt_dic

if __name__=='__main__': 
    # =============================================================================
    #   calculate CRPSS    
    # =============================================================================
    ## load daytimeindex and timestamps of targets in test data
    # testData = np.load('data/NSRDB_lag0d_lt0_pred1d/2018_NSRDB_lag0d_lt0_pred1d_TEST.npz') 
    # ## load persistence model results
    # persisDF = pd.read_csv('data/normalized_persistence_results_v2.csv')
    # # load model predictions
    # mdl_data = np.load('test/'+'prediction_NSRDB_lag0d_lt0_pred1d_pinball_regloss__2023-07-27_15-50-51_wVD'+
    #                     '/predictions/best_predictions.npz')
    # # daily_crpss = CRPSS(testData,persisDF,mdl_data,scale='daily')
    # hourly_crpss, mdl_crps,persis_crps, timestamps = CRPSS(testData,persisDF,mdl_data,scale='hourly')
    # df = pd.DataFrame({'TimeStamps':timestamps,'Hourly_CRPSS':hourly_crpss})
    # df.to_csv('results/Hourly_CRPSS_of_PV_LOAD_NL_lag24_lt24_regweight7.csv',index=False)
    # ###plot hourly crpss
    # df['TimeStamps'] = pd.to_datetime(df['TimeStamps'])
    # df['Hour'] = df['TimeStamps'].dt.hour
    # sumryhrs = df.groupby(['Hour'])['Hourly_CRPSS'].agg(lambda x: x.unique().sum()/x.nunique()) 
    # plotSingleVar(sumryhrs.to_numpy(),'Hours','CRPSS',xsticks=sumryhrs.index.to_numpy())
    # ###plot day of week crpss
    # df['dayofweek'] = df['TimeStamps'].dt.dayofweek
    # sumrydays = df.groupby(['dayofweek'])['Hourly_CRPSS'].agg(lambda x: x.unique().sum()/x.nunique()) 
    # plotSingleVar(sumrydays.to_numpy(),'Day of week','CRPSS',xsticks=sumrydays.index.to_numpy())
    # ### plot daily crps
    # plotSingleVar(daily_crpss,'Days','CRPSS')
    # =============================================================================
    #     calculate measurements of multiple predictions
    # ============================================================================= 
    expPath = "competition_experiments/" 
    dataFolders = ["Amity_competition_lag0d_lt0d_pred1d_daytime",
                   "Donalsonville_competition_lag0d_lt0d_pred1d_daytime",
                   "San_Antonio_competition_lag0d_lt0d_pred1d_daytime",
                   "Waianae_competition_lag0d_lt0d_pred1d_daytime"] 
    strategy = ['','narrow','wide'] 
    expfiles = os.listdir(os.getcwd()+'/'+expPath)
    pred_folders = []
    test_Path = []
    ### find folders of predictions, eg., prediction_NL_lag_1d_lt24_pinball_regloss__2023-07-25_22-34-43_de3
    for i in range(len(dataFolders)):
        for j in range(len(strategy)):
            pred_str = 'prediction_'+ dataFolders[i]+"_pinball_regloss_"+strategy[j]+'_'
            filename = [fn for fn in expfiles if pred_str in fn]
            pred_folders.append(filename[0])
            test_Path.append(dataFolders[i]+'/'+dataFolders[i]+'_TEST.npz')
            
    rslt_dic = collectLoss(expPath, test_Path, pred_folders)
    df = pd.DataFrame(rslt_dic)
    df.to_csv('results/comparisons_competition_lag0d_lt0d_pred1d_pinball_regloss_daytime.csv',index=False)
