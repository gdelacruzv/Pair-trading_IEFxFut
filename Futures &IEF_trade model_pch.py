# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 23:13:39 2023

@author: Gilberto
"""

import pandas as pd
import yfinance as yf
from fredapi import Fred
import statsmodels.formula.api as smf
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
%matplotlib inline

data = pd.DataFrame()
data['IEF'] = yf.download("IEF", start="2000-01-01", end="2023-10-06")['Close']
data['ZN'] = yf.download("ZN=F", start="2000-01-01", end="2023-10-06")['Close']

data['IEF_pch'] = data['IEF'].pct_change() *100
data['ZN_pch'] = data['ZN'].pct_change() *100
# Create an empty DataFrame

#Clean up Nan Values
data.dropna(inplace=True)
# Splitting the dataset
cutoff_date = '2012-01-01'

train_data = data[:cutoff_date].copy()
test_data = data[cutoff_date:].copy()

#OLS Model
Spread_model = smf.ols(formula='IEF_pch ~ ZN_pch ', data=train_data).fit()
print(Spread_model.summary())


# Get the residuals
train_data['residuals'] = Spread_model.resid

# Fit ARIMA(1,1,2) model on the residuals
arima_model = ARIMA(train_data['residuals'], order=(1,1,2)).fit()
print(arima_model.summary())
# Get the in-sample predicted residuals
train_data['predicted_residual'] = arima_model.fittedvalues
# Get the residuals
train_data['AR_residuals'] = arima_model.resid
# Get Residuals for OOS test

# Predict 'IEF' for test_data using OLS model
test_data['predicted_IEF'] = Spread_model.predict(test_data)
# Calculate residuals for the test data
test_data['residuals'] = test_data['IEF_pch'] - test_data['predicted_IEF']
arima_model_test = ARIMA(test_data['residuals'], order=(1,1,2)).fit()


# Forecast residuals using ARIMA model for the length of test_data




# Plot the residuals
plt.figure(figsize=(10,6))
plt.plot(train_data.index, train_data['predicted_residual'], label='Residuals')
plt.title('Residuals from the OLS Regression')
plt.xlabel('Date')
plt.ylabel('Residual Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

sm.qqplot(train_data['residuals'], line='s')  # 's' indicates a standardized line 
plt.title('Normal Q-Q plot of residuals')
plt.show()

#OOS Residuals plot
plt.figure(figsize=(10,6))
plt.plot(test_data.index, test_data['residuals'], label='OOS Residuals', color='blue')
plt.title('Out-of-Sample Residuals ')
plt.xlabel('Date')
plt.ylabel('Residual Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#Generate Signals
MIN_RESIDUAL = 0.05  # This is an example threshold. Adjust based on your strategy.

train_data['Signal'] = 0
train_data.loc[train_data['residuals'] > MIN_RESIDUAL, 'Signal'] = 1
train_data.loc[train_data['residuals'] < -MIN_RESIDUAL, 'Signal'] = -1

MIN_RESIDUAL_os = 0.1
#Generate signals Test
test_data['Signal'] = 0
test_data.loc[test_data['residuals'] > MIN_RESIDUAL_os, 'Signal'] = 1
test_data.loc[test_data['residuals'] < -MIN_RESIDUAL_os, 'Signal'] = -1



# 1. Calculate individual returns
def compute_portfolio_returns(data):
    # 1. Calculate individual returns
    data['IEF_returns'] = data['IEF'].pct_change().shift(-1)
    data['ZN_returns'] = data['ZN'].pct_change().shift(-1)
    
    # 2. Compute portfolio returns based on the signal
    data['portfolio_returns'] = 0.0

    # When Signal is 1 (Long Spread: Buy IEF & Short ZN)
    data.loc[data['Signal'] == -1, 'portfolio_returns'] = data['IEF_returns'] - data['ZN_returns']

    # When Signal is -1 (Short Spread: Short IEF & Buy ZN)
    data.loc[data['Signal'] == 1, 'portfolio_returns'] = -data['IEF_returns'] + data['ZN_returns']
    
    
    return data

# Now you can call this function for either train_data or test_data:
train_data = compute_portfolio_returns(train_data)
test_data = compute_portfolio_returns(test_data)
# or
# The 'portfolio_returns' column now has the daily return of the spread trading strategy.

def plot_performance(data): 
   #Visualize the performance:
    plt.figure(figsize=(10, 6))
    plt.plot(data['portfolio_returns'], label="Strategy")
    plt.legend()
    plt.show()

plot_performance(train_data)
plot_performance(test_data)


def portfolio_metrics(data):
    # Sharpe Ratio
    risk_free_rate = 0.01  
    daily_risk_free_rate = (1 + risk_free_rate) ** (1/252) - 1
    excess_daily_return = data['portfolio_returns'] - daily_risk_free_rate
    sharpe_ratio = excess_daily_return.mean() / excess_daily_return.std() * (252 ** 0.5)

    # Maximum Drawdown
    data['Cumulative Wealth'] = (1 + data['portfolio_returns']).cumprod()
    running_max = data['Cumulative Wealth'].cummax()
    drawdown = data['Cumulative Wealth'] / running_max - 1
    max_drawdown = drawdown.min()
    
    # Total Portfolio Return
    total_return = data['Cumulative Wealth'].dropna().iloc[-1] - 1

    # Win/Loss Metrics
    wins = data['portfolio_returns'][data['portfolio_returns'] > 0]
    losses = data['portfolio_returns'][data['portfolio_returns'] < 0]

    win_loss_ratio = len(wins) / len(losses) if len(losses) != 0 else float('inf')
    average_win = wins.mean()
    average_loss = losses.mean()

    print(f"Sharpe Ratio: {sharpe_ratio}")
    print(f"Maximum Drawdown: {max_drawdown}")
    print(f"Win/Loss Ratio: {win_loss_ratio}")
    print(f"Average Win: {average_win}")
    print(f"Average Loss: {average_loss}")
    print(f"Total Portfolio Return: {total_return * 100:.2f}%")
# Now you can call this function for either train_data or test_data:
portfolio_metrics(train_data)
# or
portfolio_metrics(test_data)

# For this, you need to create a small DataFrame for the next 3 periods using test_data

# Forecast the next 3 residuals using ARIMA model
# Forecast the next 3 residuals using the ARIMA model trained on test_data residuals
forecasted_resid = arima_model_test.forecast(steps=3)

# Create a new DataFrame for the forecasted residuals
forecasted_dates = pd.date_range(test_data.index[-1] + pd.Timedelta(days=1), periods=3, freq='D')
next_periods = pd.DataFrame({
    'forecasted_residuals': forecasted_resid.values
}, index=forecasted_dates)

print(next_periods[['forecasted_residuals']])