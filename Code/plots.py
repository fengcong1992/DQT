import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go

filenames = ['prediction_NSRDB_lag0_lt0_pred1d_daytime_only_pinball_regloss__2023-08-07_20-41-01_YLL',
            '',
            '']
 
data = np.load('test/'+filenames[0]+'/predictions/best_predictions.npz')
predictions = data['predictions']
targets = data['targets']
targets = np.reshape(targets,(-1,1))
predictions = np.sort(predictions, axis=2)
predictions = np.reshape(predictions,(-1,11))
timeStamps = np.load('data/NSRDB_lag0_lt0_pred1d_daytime_only/2018_NSRDB_lag0_lt0_pred1d_daytime_only_TEST.npz')['targetsTime']
# timeStamps = np.reshape(timeStamps,(-1,1))
# timeStamps = timeStamps.astype('datetime64')
# df = pd.read_csv('data/normalized_persistence_results_v2.csv')
# df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
# df = df[df['TimeStamp']>pd.Timestamp(year = 2018,month=1,day=1,hour=10)]
config = {
  'toImageButtonOptions': {
    'format': 'png', # one of png, svg, jpeg, webp
    'filename': 'custom_image',
    'height': 200,
    'width': 300,
    'scale':5 # Multiply title/legend/axis/canvas sizes by this factor
  }
}

df = pd.DataFrame(predictions,columns=['P0','P1','P2','P3','P4','P5','P6','P7','P8','P9','P10'])
df['TimeStamp'] = np.reshape(timeStamps,(-1,1))
df['targets'] = targets

fig = go.Figure()

for i in range(11):
        fig.add_trace(go.Scatter(name=f'P{i}', x = df['TimeStamp'], y = df[f'P{i}']))

fig.add_trace(go.Scatter(name='targets', x = df['TimeStamp'], y = df['targets'], 
                         mode = 'lines+markers',
                          marker=dict(
            color='Black',
            size=8,
            line=dict(color='Black',
                width=1))))

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
fig.write_html('NSRDB_lag0_lt0_pred1d_daytime_only_figure.html', auto_open=True)
 
# filename = 'prediction_pinball_loss_HECO_nlag24_lt24_2023-07-06_11-41-15_7EN'

# data = sio.loadmat('experiments/'+filename+'/predictions/best_predictions.mat')
# predictions = data['predictions']
# targets = data['targets']
# targets = np.reshape(targets,(-1,1))
# predictions = np.sort(predictions, axis=1)

# with pd.ExcelWriter('experiments/'+filename+'/pinball_loss_HECO_nlag24_lt24_NoNetLoad.xlsx') as ew:
#     df = pd.DataFrame(data = predictions,columns=['P0','P1','P2','P3','P4','P5','P6','P7','P8','P9','P10'])
#     df['targets'] = targets 
#     df.to_excel(ew,sheet_name='Sheet1')
    

# # plt.figure(dpi=300) 
# fig, ax = plt.subplots()  
# ax.plot(np.linspace(1, 49,48),targets[0:48,:], 'b-*',linewidth=2.0, label='targets')
# for i in range(11):
#     ax.plot(np.linspace(1, 49,48),predictions[0:48,i], linewidth=1.0,label = 'P'+str(i))
# ax.legend()
# plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
# plt.xlabel("Hours")
# plt.ylabel("Normalized Net Load")
# plt.show()
# plt.savefig('HECO_nlag24_lt24_withNL.png', dpi=300)