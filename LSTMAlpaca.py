#Importing libraries
import tensorflow
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


#Get data from Alpaca using my key
client = StockHistoricalDataClient(api_key="INSERT KEY HERE", secret_key="INSERT KEY HERE")

request_params = StockBarsRequest(
    symbol_or_symbols=["RKLB"],
    timeframe=TimeFrame.Day,
    start=datetime(2021, 8, 25),
    end = datetime(2025, 6, 30)
)

bars = client.get_stock_bars(request_params)

# Convert to DataFrame
df = bars.df.reset_index()

# Initial Data Exploration (head, info, describe)
print("Head of DataFrame:")
print(df.head())

print("\n DataFrame Info:")
print(df.info())

print("\n Descriptive Statistics")
print(df.describe())

# Initial Data Visualisation
plt.figure(figsize=(12,6))
plt.plot(df['timestamp'], df['close'], label='Close Price')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.title('RKLB Close Prices Over Time')
plt.legend()
plt.show()

#Get numeric data from df
data = df.select_dtypes(include=[np.number])

#Plot correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), annot=True)
plt.title('Feature Correlation Heatmap')
plt.show()

#Convert data into datetime
data["date"] = df["timestamp"]

prediction = data.loc[ (data["date"] > "2021-08-25") & (data["date"] < "2025-6-30")]

#Creating sequential LSTM

#Gather training data and format for TF
dataset = data["close"].values
training_data_len = int(np.ceil(0.95 * len(dataset)))

scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset.reshape(-1, 1))
training_data = scaled_data[:training_data_len]

x_train = []
y_train = [] 

for i in range(60, len(training_data)):
    x_train.append(training_data[i-60:i, 0])
    y_train.append(training_data[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Build LSTM model
model = keras.Sequential()
#1st layer
model.add(keras.layers.LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))

#2nd layer
model.add(keras.layers.LSTM(128, return_sequences=False,))

#3rd layer
model.add(keras.layers.Dense(128, activation="relu"))

#4th layer
model.add(keras.layers.Dropout(0.5))

#Output layer
model.add(keras.layers.Dense(1))

#Compile and train the model
model.summary()
model.compile(optimizer="adam", loss="mean_squared_error", metrics=[keras.metrics.RootMeanSquaredError()])

training = model.fit(x_train, y_train, epochs=50, batch_size=32)
test_data = scaled_data[training_data_len - 60:]
x_test = []
y_test = []
dataset[training_data_len:]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Making a prediction
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

train = data[:training_data_len]
test = data[training_data_len:]

test = test.copy()

test["Predictions"] = predictions
 
#Plotting data
plt.figure(figsize=(12,6))
plt.plot(train["date"], train["close"], label="Training data", color="blue")
plt.plot(test["date"], test["close"], label="Test data", color="orange")
plt.plot(test["date"], test["Predictions"], label="Predictions", color="red")
plt.title("LSTM Model Predictions")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

#Plotting final 5% of data
plt.figure(figsize=(12,6))
plt.plot(test["date"], test["close"],    label="Actual Price",    color="orange")
plt.plot(test["date"], test["Predictions"], label="Predicted Price", color="red")
plt.title("LSTM – Test Set Only")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# Finding the MSE and MAE
y_true = test["close"].values
y_pred = test["Predictions"].values
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

#Get R² for train and test
train_preds = model.predict(x_train)
train_preds = scaler.inverse_transform(train_preds).flatten()
train_true = data["close"].values[60:training_data_len]

#Get MSE and MAE for train and test
train_preds = model.predict(x_train)
train_preds = scaler.inverse_transform(train_preds).flatten()
train_true = data["close"].values[60:training_data_len]

#Output Train and Test R², MSE and MAE
train_mse = mean_squared_error(train_true, train_preds)
train_mae = mean_absolute_error (train_true, train_preds)
print(f"Train   MSE: {train_mse:.4f}, MAE: {train_mae:.4f}")

test_mse = mean_squared_error(y_true, y_pred)
test_mae = mean_absolute_error (y_true, y_pred)
print(f" Test   MSE: {test_mse:.4f}, MAE: {test_mae:.4f}")

train_r2 = r2_score(train_true, train_preds)
print(f"Train   R²: {train_r2:.4f}")

test_r2 = r2_score(y_true, y_pred)
print(f" Test   R²: {test_r2:.4f}")
