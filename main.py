import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Load data --- 
company = input("Enter the stock/crypto symbol (e.g., BTC-USD) (from yfinance): ")
data = yf.download(company, period="60d", interval="1h", progress=False)

close_prices = data["Close"].values.reshape(-1, 1)

# --- Separate last 48 points as future test set ---
future_steps = 48
train_prices = close_prices[:-future_steps]
test_prices = close_prices[-future_steps:]


# --- Scale Data ---
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(train_prices)

train_data_amount = len(scaled_train)
# Block 1
x_train_1 = scaled_train[0:train_data_amount // 3, 0].reshape(1, train_data_amount // 3, 1)
y_train_1 = np.array([scaled_train[train_data_amount // 3 - 1, 0]])

# Block 2
x_train_2 = scaled_train[0:train_data_amount // 2, 0].reshape(1, train_data_amount // 2, 1)
y_train_2 = np.array([scaled_train[train_data_amount // 2 - 1, 0]])

# Block 3
x_train_3 = scaled_train[0:train_data_amount, 0].reshape(1, train_data_amount, 1)
y_train_3 = np.array([scaled_train[-1, 0]])

# Block 4
x_train_4 = scaled_train[train_data_amount - (train_data_amount // 3):train_data_amount, 0].reshape(1, train_data_amount // 3, 1)
y_train_4 = np.array([scaled_train[-1, 0]])


# --- Build Model ---
model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(None, 1)))  
model.add(Dropout(0.4))
model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=128))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error")

# --- Train progressively ---
model.fit(x_train_1, y_train_1, epochs=32, batch_size=16, sample_weight=np.array([1.0]))
model.fit(x_train_2, y_train_2, epochs=32, batch_size=16, sample_weight=np.array([2.0]))
model.fit(x_train_3, y_train_3, epochs=32, batch_size=16, sample_weight=np.array([3.0]))
model.fit(x_train_4, y_train_4, epochs=32, batch_size=16, sample_weight=np.array([4.0]))

# --- test model ---
predicted_test = []
window_size = 96
last_window = scaled_train[-window_size:].reshape(1, window_size, 1)

for i in range(48):
    pred = model.predict(last_window, verbose=0)        
    predicted_test.append(pred[0, 0])                   
    last_window = np.append(last_window[:, 1:, :], [[[pred[0, 0]]]], axis=1)

predicted_test = scaler.inverse_transform(np.array(predicted_test).reshape(-1, 1))


# --- plot ---
plt.plot(test_prices, color="black", label=f"Actual {company} Price")
plt.plot(predicted_test, color="green", label=f"Predicted {company} Price")
plt.title(f"{company} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{company} Share Price")
plt.legend()
plt.show()

# -- analysiation of the model --- 
mae = mean_absolute_error(test_prices, predicted_test)
rmse = np.sqrt(mean_squared_error(test_prices, predicted_test))
r2 = r2_score(test_prices, predicted_test)

print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)
