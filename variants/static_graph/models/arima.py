from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import ParameterGrid
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

from utils import rmse

def arima_forecast(df):
    # Create a 'ds' column with a uniform date time period
    start_date = datetime.today() - timedelta(days=len(df))
    df['ds'] = pd.date_range(start=start_date, periods=len(df), freq='D')

    # Split the dataframe into train and test sets
    train_size = int(len(df) * 0.8)
    train, test = df[0:train_size], df[train_size:len(df)]
    
    # Initialize a dictionary to store the models for each column
    models = {}

    # Define the grid of hyperparameters to search
    hyperparameters = {'p': range(0, 2), 'd': range(0, 2), 'q': range(0, 2)}

    # Fit a model for each column in the dataframe
    for column in train.columns:
        if column != 'ds':
            best_score, best_params = float("inf"), None
            # Grid search
            for params in ParameterGrid(hyperparameters):
                try:
                    model = ARIMA(train[column], order=(params['p'], params['d'], params['q']))
                    model_fit = model.fit()
                    rmse_val = rmse(train[column], model_fit.fittedvalues)
                    if rmse_val < best_score:
                        best_score, best_params = rmse_val, params
                except:
                    continue
            # Refit the model with the best parameters found
            model = ARIMA(train[column], order=(best_params['p'], best_params['d'], best_params['q']))
            model_fit = model.fit()
            models[column] = model_fit
    
    # Initialize a dictionary to store the forecasts
    forecasts = {}

    # Generate a forecast for each column
    for column in test.columns:
        if column != 'ds':
            forecast = models[column].forecast(steps=len(test))
            forecasts[column] = forecast

    # Calculate the RMSE for each column and return the average
    rmse_values = []
    for column in test.columns:
        if column != 'ds':
            rmse_val = rmse(test[column], forecasts[column])
            rmse_values.append(rmse_val)

    avg_rmse = np.mean(rmse_values)
    return avg_rmse, train, test, forecasts