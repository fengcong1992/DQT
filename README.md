# Deep Quantile Transfomer
This is the code for our project, titled Deep Quantile Transformer for Day-ahead Probabilistic Net Load Forecasting
## Rerequisites and Environment
Refer to ```requirements.txt``` for the environment setup
## Project/Repository Structure
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
## Data Collection
1. Weather actuals/forecasts could be obtained from open-source database, such as the ECMWF by using [Open-Meteo](https://open-meteo.com). Example see ```download_data.py```
2. Net load actual data is provided by the [American-Made Net Load Forecasting Prize](https://www.energy.gov/eere/solar/american-made-net-load-forecasting-prize)
3. Persistence ensemble forecasts are generated using ```PE.py``` following the [Prize Competition rules](https://americanmadechallenges.org/challenges/net-load-forecasting/docs/Net-Load-Forecasting-Rules.pdf)
## Experiments
Things to change based on your own folder structure:
1.
2.
3.

Prepare the data for training and testing by running ```sample_data_process.py```

Train DQT models for the four locations and generate daily net load forecasts using ```main.py```


## Evaluation
The CRPS and CRPSS are used to used to evaluate the probabilistic net load forecasts. The DQT performances are listed in the following table and could be compared with the top teams in the [competition](https://www.herox.com/net-load-forecasting/leaderboard)
| Location  | PE CRPS | DQT CRPS | DQT CRPSS|
| --------- | ------- | -------- | -------- |
| OR  | 0.0414  | 0.0290 | 0.2995|
| GA  | 0.0475  | 0.0436 | 0.0821 |
| TX  | 0.0458  | 0.0368 | 0.1965 |
| HI  | 0.0848  | 0.0850 | -0.0024 |
| Ave.| 0.0549  | 0.0486 | 0.1927 |
