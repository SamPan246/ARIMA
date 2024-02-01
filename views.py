
from django.shortcuts import render
from django.http import HttpResponse
import os
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from matplotlib.backends.backend_agg import FigureCanvasAgg 
from django.http import HttpResponseRedirect
#from django.shortcuts import requests

from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

import time
import datetime
import calendar

from django.urls import path
from . import views






# Create your views here.
# In the cell below, we look up a stock, compute the epoch time for today, and query the data on the last five years according to yahoo.com."""
def ok(request):
    return render(request,'index.html')

def readInput(request):
    stock_name = request.POST.get('stock_name')
    print(stock_name)
    return True

stock_name="PXD"
t = int(datetime.datetime.now().timestamp())
dateparse = lambda dates: pd.to_datetime(dates, format='%Y-%m-%d')
stock_data = pd.read_csv('https://query1.finance.yahoo.com/v7/finance/download/'+stock_name+'?period1='+str(t-157766400)+'&period2='+str(t)+'&interval=1d&events=history&includeAdjustedClose=true',sep=',', index_col='Date', parse_dates=['Date'], date_parser=dateparse).fillna(0)

    
df_close = stock_data['Close']
df_log = np.log(df_close)
train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]

model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0,
                      D=0,
                      trace=False,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)

model = ARIMA(train_data, order=model_autoARIMA.order, seasonal_order=(1,1,1,5))
fitted = model.fit()

stuff = fitted.get_forecast(len(test_data), alpha=0.05)  # 95% conf
stuff_df = stuff.summary_frame()
fc = list(stuff_df['mean'])
se = list(stuff_df['mean_se'])
conf = stuff_df[['mean_ci_lower','mean_ci_upper']]
# Make as pandas series
fc_series = pd.Series(fc, index=test_data.index)
lower_series = pd.Series(list(conf.iloc[:, 0]), index=test_data.index)
upper_series = pd.Series(list(conf.iloc[:, 1]), index=test_data.index)
# Plot

fig, ax = plt.subplots()
fig.set_size_inches(10,5)
ax.plot(train_data, label='training data')
ax.plot(test_data, color = 'blue', label='Actual Price')
ax.plot(fc_series, color = 'orange',label='Predicted Price')
ax.fill_between(lower_series.index, lower_series, upper_series,
                 color='k', alpha=.10)
ax.set_title(stock_name+' Price Prediction')
ax.set_xlabel('Time')
ax.set_ylabel('Log '+stock_name+' Price')
ax.legend(loc='upper left', fontsize=8)
response = HttpResponse(content_type = 'image/png')
canvas = FigureCanvasAgg(fig)
canvas.print_png(response)
my_path = os.path.abspath("ArimaApp")
fig.savefig(my_path + '/static/image.png')
   
   #def ok1(request):
 #return render(request, "index.html", {"stock_name":stock_name})
# report performance
mse = mean_squared_error(test_data, fc)
print('MSE: '+str(mse))
mae = mean_absolute_error(test_data, fc)
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(test_data, fc))
print('RMSE: '+str(rmse))
mape = np.mean(np.abs(fc - test_data)/np.abs(test_data))
print('MAPE: '+str(mape))

mape_final = (mape * 100)

accuracy = (100 - mape_final)
