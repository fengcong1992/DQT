#from openmeteo_py.Daily.Marine import Marine as Daily
from openmeteo_py import OWmanager
from openmeteo_py.Hourly.HourlyHistorical import HourlyHistorical
from openmeteo_py.Daily.DailyHistorical import DailyHistorical
from openmeteo_py.Options.HistoricalOptions import HistoricalOptions  
from openmeteo_py.Hourly.HourlyGfs import HourlyGfs
from openmeteo_py.Daily.DailyGfs import DailyGfs
from openmeteo_py.Options.GfsOptions import GfsOptions 
from openmeteo_py.Utils.constants import *

# Latitude, Longitude  
cities = {'city':['Amity','Donalsonville','SanAntonio','Waianae'], 'latitude':[45.114559,31.044241,29.424122,21.446911],
          'longtitude':[-123.204903,-84.879128,-98.493629,-158.188736]}

### historical data 
start_date = "2022-01-01"
end_date = "2023-07-19"

for i in range(4):
    city = cities['city'][i] 
    # =============================================================================
    #     ### Get historical data
    # =============================================================================
    hourly = HourlyHistorical()
    daily = DailyHistorical()  
    options = HistoricalOptions(cities['latitude'][i],cities['longtitude'][i],nan,False,
                                celsius,kmh,mm,iso8601,utc,start_date,end_date)  
    mgr = OWmanager(options,OWmanager.historical,hourly.all()) 
    meteo_hist = mgr.get_data(3)

    # =============================================================================
    #     ### Get NOAA GFS data
    # =============================================================================
    hourly = HourlyGfs()
    daily = DailyGfs()
    options = GfsOptions(cities['latitude'][i],cities['longtitude'][i],start_date=start_date,
                          end_date=end_date)
    mgr = OWmanager(options,OWmanager.gfs,hourly.all()) 
    meteo_gfs = mgr.get_data(3)
    
    # =============================================================================
    #     # compare historical features and GFS forecasting features
    #     # drop inconsistent features
    # =============================================================================
    hist_cols = set(meteo_hist.columns)
    gfs_cols = set(meteo_gfs.columns) 
    not_in_gfs_fts = [ele for ele in hist_cols if ele not in gfs_cols]
    meteo = meteo_hist.drop(not_in_gfs_fts,axis=1)
    # =============================================================================
    #     ### drop columns filled with 0
    # ============================================================================= 
    meteo = meteo.drop(['snowfall','precipitation','weathercode','cloudcover_low',
                        'cloudcover_mid','cloudcover_high'],axis=1)
    
    # =============================================================================
    #     ### save file
    # =============================================================================
    meteo = meteo.rename(columns={"time": "TimeStamp"})
    filename = f'{city}_{start_date}_{end_date}_hourly_historical_data.csv'
    meteo.to_csv('data/'+filename, index=False)
    