# measure persistence model daytime performances
import pandas as pd
import numpy as np
from models.measurements import measurements
from pysolar.solar import *
import csv

time_zone = {'city':['Amity','Donalsonville','San_Antonio','Waianae'],
             'start_time':[8,5,6,11],'latitude':[45.114559,31.044241,29.424122,21.446911],
                       'longtitude':[-123.204903,-84.879128,-98.493629,-158.188736]}

rslt_dic = {'experiment_name':[],'CRPS':[],'CRPS_sorted':[],'CRPS_dayloss':[],\
            'CRPS_dayloss_sorted':[],'pinball_loss':[],'pinball_loss_sorted':[]}
###Add zenith
for icity in range(4): 
    data = pd.read_csv(f"data/{time_zone['city'][icity]}_persistence_model_results.csv")
    dt = pd.to_datetime(data['TimeStamp'])
    
    ### add zenith according to time
    zenith = []
    for i in range(len(dt)):
        idt = dt[i].to_pydatetime() 
        al = get_altitude(time_zone['latitude'][icity], time_zone['longtitude'][icity], idt)
        zenith.append(90-al)
    data['Zenith'] = zenith
    
    daytime = np.array([1]*len(data))
    idx = data[data['Zenith']>87].index
    daytime[idx] = 0 
    data['daytimeIdx'] = daytime
    # data.to_csv(f"data/{time_zone['city'][icity]}_persistence_model_results.csv", index=False)
    
    rslt_dic['experiment_name'].append(f"{time_zone['city'][icity]}_persistence_model_results")
    pred = np.reshape(data.iloc[:,1:12].to_numpy(),(-1,24,11))
    tar = np.reshape(data['targets'].to_numpy(),(-1,24))
    daytimeIdx = np.reshape(data['daytimeIdx'].to_numpy(),(-1,24))
     
    loss = measurements(pred, tar, regweight=7)
    rslt_dic['CRPS'].append(loss.CRPS_loss())
    rslt_dic['pinball_loss'].append(loss.pinball_loss())
    rslt_dic['CRPS_dayloss'].append(loss.CRPS_dayloss(daytimeIdx)) 
    pred = np.sort(pred,axis=2)
    rslt_dic['CRPS_sorted'].append(loss.CRPS_loss()) 
    rslt_dic['pinball_loss_sorted'].append(loss.pinball_loss())  
    rslt_dic['CRPS_dayloss_sorted'].append(loss.CRPS_dayloss(daytimeIdx))
    
rslt_dic = pd.DataFrame(rslt_dic)
rslt_dic.to_csv('results/Competition_persistence_models_measurements.csv',index=False)

# df = pd.read_csv('data/Amity_persistence_model_results.csv')
# heco_df = pd.read_excel('data/HECOData_Clean_2018_with_zenith.xlsx')

# newdf = pd.merge(df, heco_df, how = 'inner', on='TimeStamp')
# daytime = np.array([1]*len(newdf))
# idx = newdf[newdf['Zenith']>87].index
# daytime[idx] = 0 
# newdf['daytimeIdx'] = daytime

# rslt_dic = {'experiment_name':[],'CRPS':[],'CRPS_sorted':[],'CRPS_dayloss':[],\
#             'CRPS_dayloss_sorted':[],'pinball_loss':[],'pinball_loss_sorted':[]}
# pred = np.reshape(newdf.iloc[:,1:12].to_numpy(),(-1,24,11))
# tar = np.reshape(newdf['targets'].to_numpy(),(-1,24))
# daytimeIdx = np.reshape(newdf['daytimeIdx'].to_numpy(),(-1,24))


# loss = measurements(pred, tar, regweight=7)
# rslt_dic['experiment_name'].append('persistence_model_08012018') 
# rslt_dic['CRPS'].append(loss.CRPS_loss())
# rslt_dic['pinball_loss'].append(loss.pinball_loss())
# rslt_dic['CRPS_dayloss'].append(loss.CRPS_dayloss(daytimeIdx)) 
# pred = np.sort(pred,axis=2)
# loss = measurements(pred, tar, regweight=5) 
# rslt_dic['CRPS_sorted'].append(loss.CRPS_loss()) 
# rslt_dic['pinball_loss_sorted'].append(loss.pinball_loss())
# rslt_dic['CRPS_dayloss_sorted'].append(loss.CRPS_dayloss(daytimeIdx)) 
# df = pd.DataFrame(rslt_dic)
# df.to_csv('results/persistence_model_measurements.csv',index=False)

