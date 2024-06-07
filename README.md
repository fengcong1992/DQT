# Deep Quantile Transfomer
This is the code for our project, titled Deep Quantile Transformer for Day-ahead Probabilistic Net Load Forecasting
# Rerequisites and environment
Refer to ```requirements.txt``` for the environment setup
# Project/Repository Structure
```
|-- code
|-- data
|   |-- PE forecasts
|   |-- net load
|   |-- weather forecasts
|-- model
|-- output
|   |-- ga
|   |-- hi
|   |-- or
|   |-- tx
```
# Data Collection
1. Weather actuals/forecasts could be obtained from open-source database, such as the ECMWF by using [Open-Meteo](https://open-meteo.com). Example see ```download_data.py```
2. Net load actual data is provided from the [American-Made Net Load Forecasting Prize](https://www.energy.gov/eere/solar/american-made-net-load-forecasting-prize)
3. Persistence ensemble forecasts are generated using ```PE.py``` following the Prize Competition rules
# Experiments
Train DQT models for the four locations and generate daily net load forecasts using ```DQT_MainScript.py```
# Evaluation
| Location  | PE CRPS | DQT CRPS | DQT CRPSS|
| --------- | ------- | -------- | -------- |
| OR  | Content Cell  |
| GA  | Content Cell  |
| OR  | Content Cell  |
| GA  | Content Cell  |
