import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# d1 = pd.read_excel("data/HECOData_Clean_2017_with_zenith.xlsx")
# d2 = pd.read_excel("data/HECOData_Clean_2018_with_zenith.xlsx")

# df = pd.concat([d1,d2],ignore_index=True)
# daytime = df['Zenith'].to_numpy()
# daytime = np.where(daytime>87,0,1)

d1 =np.load("data/Waianae_competition_lag0d_lt0d_pred1d/Waianae_competition_lag0d_lt0d_pred1d_TRAIN.npz",allow_pickle=True)
d2 =np.load("data/Waianae_competition_lag0d_lt0d_pred1d/Waianae_competition_lag0d_lt0d_pred1d_TEST.npz",allow_pickle=True)
dt1 = d1['daytimeIdx']
dt2 = d2['daytimeIdx']
daytimeIdx = np.concatenate((dt1,dt2),axis=0)
targetsTime = np.concatenate((d1['targetsTime'],d2['targetsTime']),axis=0)
daytimeIdx = daytimeIdx.flatten()
targetsTime = targetsTime.flatten()
# s = np.sum(daytime, axis =0)

# fig = go.Figure()
# fig.add_trace(go.Histogram(histfunc="count",  x=s))
# fig.show()
# fig.write_html('Amity_duration_count_figure.html', auto_open=True)

timestamps = [v for i,v in enumerate(targetsTime) if daytimeIdx[i]==1]
timestamps = pd.to_datetime(timestamps)
hours = timestamps.hour

new_df = pd.DataFrame(hours,columns=['hours'])
new_df['timestamps'] = timestamps

(new_df.groupby('hours')
    ['timestamps'].nunique()
    .plot.bar())