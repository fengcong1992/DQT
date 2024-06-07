import pandas as pd
import numpy as np
import os

root_obs = 'your directory'
root_save = 'your directory'
day_pe = 30 # previous 30 days of same hours
list_hour = list(range(24))
list_percentiles = list(np.arange(0,110,10))
list_loc = ['OR', 'GA', 'TX', 'HI']
i_loc = 0
for loc in list_loc:
    list_files = [x for x in os.listdir(root_obs) if loc in x]
    # read and combine data
    df_obs1 = pd.read_csv(os.path.join(root_obs, list_files[0]), encoding="utf-8", header=2)  # has utc offset
    df_obs2 = pd.read_csv(os.path.join(root_obs, list_files[1]), encoding="utf-8", header=2)  # has utc offset
    df_obs = pd.concat([df_obs1, df_obs2], axis=0)
    df_obs = df_obs.sort_values(by='timestamp')

    # preprocess
    df_obs['timestamp'] = pd.to_datetime(df_obs['timestamp'])

    # get days to forecast
    list_date = pd.date_range(start=df_obs['timestamp'].dt.date.min(), end=df_obs['timestamp'].dt.date.max(), freq='1D') #df_obs['timestamp'].dt.date.unique()
    df_all = pd.DataFrame()
    for date in list_date[day_pe:]:
        print('PE Forecasting:', date)
        for hour in list_hour:
            df_date_hour = df_obs[(df_obs['timestamp'] < pd.to_datetime(date, utc=True)) & 
                                (df_obs['timestamp'] >= pd.to_datetime(date-pd.DateOffset(days=30), utc=True)) &
                                (df_obs['timestamp'].dt.hour == hour)]
            percentiles = np.nanpercentile(a=df_date_hour['value'], q=list_percentiles)
            df = pd.DataFrame([np.concatenate(([date+pd.Timedelta(hours=hour)], percentiles))], columns=['timestamp'] + [f'Q{i*10}' for i in range(0, len(percentiles))])
            df_all = pd.concat([df_all, df], axis=0)
    df_all.to_csv(os.path.join(root_save, 'pe_forecasts_'+loc+'.csv'), index=False)

