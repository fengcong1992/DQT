import numpy as np
import pandas as pd
from collect_results import CRPSS
import plotly.graph_objs as go

data1 = np.load('test/prediction_NSRDB_lag0d_lt0_pred1d_pinball_regloss__2023-07-27_15-50-51_wVD/predictions/best_predictions.npz')
pred = data1['predictions'] 
tar = data1['targets']
# pred = pred[0:31,:,:]
# tar = tar[0:31,:]

data2 = np.load('data/NSRDB_lag0d_lt0_pred1d/2018_NSRDB_lag0d_lt0_pred1d_TEST.npz')
startT = data2['targetsTime'][0,0]
endT = data2['targetsTime'][-1,-1]
# for key in enumerate(data2):
#     print(key)
data3 = pd.read_csv('data/normalized_persistence_results_v2.csv') 
data3['TimeStamp'] = pd.to_datetime(data3['TimeStamp'])
id1 = data3[data3['TimeStamp']==startT].index[0]
id2 = data3[data3['TimeStamp']==endT].index[0]
persis_pred = data3.iloc[id1:id2+1,:]
persis_pred['TimeStamp'] = pd.to_datetime(persis_pred['TimeStamp'])
persis_pred=persis_pred.reset_index(drop=True)

daytimeIdx = data2['daytimeIdx']
# daytimeIdx = data2['daytimeIdx'][0:31,:]
# daytimeIdx[:,:] = 1
testData = {'daytimeIdx':daytimeIdx,'targetsTime':data2['targetsTime']}
mdl_pred = {'predictions':pred,'targets':tar}
persisTime = pd.to_datetime(persis_pred['TimeStamp'])
days = pd.unique(persisTime.dt.date)
days = days[1:]
# timeStamps = testData['targetsTime']
# persisTime[persisTime==timeStamps.flatten()].index
daily_crpss,daily_mdl_CRPS,daily_persis_CRPS = CRPSS(testData,persis_pred,mdl_pred,'daily')
hourly_crpss,mdl_crps,persis_crps,timestamps = CRPSS(testData,persis_pred,mdl_pred,'hourly')
# daily_CRPSS =1-np.array(daily_mdl_CRPS)/np.array(daily_persis_CRPS)

# hours = pd.to_datetime(pd.unique(pd.to_datetime(timestamps).strftime('%m/%d/%y %H')))
hourly_mean = np.mean(hourly_crpss)
daily_mean = np.mean(daily_crpss)

config = {
  'toImageButtonOptions': {
    'format': 'png', # one of png, svg, jpeg, webp
    'filename': 'custom_image',
    'height': 200,
    'width': 300,
    'scale':5 # Multiply title/legend/axis/canvas sizes by this factor
  }
}

fig = go.Figure()
fig.add_trace(go.Scatter(name='CRPSS', x = days,y=daily_crpss))
fig.add_trace(go.Scatter(name='Model_CRPS', x = days,y=daily_mdl_CRPS))
fig.add_trace(go.Scatter(name='Persis_CRPS', x = days,y=daily_persis_CRPS))
fig.update_layout(
    plot_bgcolor='white'
)
fig.update_xaxes(
    title_text='Time',
    mirror=False,
    ticks='outside',
    showline=True,
    linecolor='black',
    gridcolor='lightgrey'
)
fig.update_yaxes(
    title_text='Normalized Net Load',
    mirror=False,
    ticks='outside',
    showline=True,
    linecolor='black',
    gridcolor='lightgrey'
)     
fig.show(config=config)
fig.write_html('daily_CRPSS_NSRDB_lag0d_lt0_pred1d_pinball_regloss_figure.html', auto_open=True)