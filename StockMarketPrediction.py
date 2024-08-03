
from turtle import st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import yfinance as yf

ticker = 'NVDA'
data = yf.download(ticker, start='2010-01-01', end='2024-08-02')

data = data[['Adj Close']]
data['Prediction'] = data['Adj Close'].shift(30)   # 30 days in future

# Remove last 30 rows
data.dropna(inplace=True)

# Features
X = data[['Adj Close']]

# Target
y = data['Prediction']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale the Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# make predictions
predictions = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, predictions)
print('Mean Squared Error: ', mse)

test_dates = data.index[-len(y_test):]

# Plot the actual vs. predicted values with dates on the x-axis
plt.figure(figsize=(12, 6))

# Plot actual prices
plt.plot(data.index[-len(y_test):], y_test.values, label='Actual Prices')

# Plot predicted prices
plt.plot(test_dates, predictions, label='Predicted Prices')

plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()