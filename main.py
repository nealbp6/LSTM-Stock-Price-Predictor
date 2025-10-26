import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Load data ---
company = input("Enter the stock/crypto symbol (e.g., BTC-USD) (from yfinance): ")
data = yf.download(company, period="180d", interval="1h", progress=False)
close_prices = data["Close"].values.reshape(-1, 1)

# --- Split data ---
future_steps = 48
train_prices = close_prices[:-future_steps]
test_prices = close_prices[-future_steps * 2:]
real_prices = close_prices[-future_steps:]

# --- Scale ---
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_prices)
scaled_test  = scaler.transform(test_prices)

# --- Prepare training data ---
window_size = 48
x_train, y_train = [], []
for i in range(window_size, len(scaled_train)):
    x_train.append(scaled_train[i - window_size:i, 0])
    y_train.append(scaled_train[i, 0])
x_train = np.array(x_train).reshape(-1, window_size, 1)
y_train = np.array(y_train)

# --- Prepare testing data ---
x_test, y_test = [], []
for i in range(window_size, len(scaled_test)):
    x_test.append(scaled_test[i - window_size:i, 0])
    y_test.append(scaled_test[i, 0])
x_test = np.array(x_test).reshape(-1, window_size, 1)
y_test = np.array(y_test)

# --- Build Model ---
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(window_size, 1)),
    Dropout(0.4),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(128),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")

# --- Train ---
n = len(y_train)
weights = np.linspace(0.5, 4.5, n)
model.fit(x_train, y_train, epochs=32, batch_size=32, sample_weight=weights, verbose=1, shuffle=False)

# --- Evaluate on test set ---
pred_scaled = model.predict(x_test, verbose=0)
pred_test = scaler.inverse_transform(pred_scaled)
y_true = scaler.inverse_transform(y_test.reshape(-1, 1))

mae = mean_absolute_error(y_true, pred_test)
rmse = np.sqrt(mean_squared_error(y_true, pred_test))
r2 = r2_score(y_true, pred_test)

print("BACKTEST RESULTS (Last 48h real data)")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²:   {r2:.3f}")

# --- Forecast next 48 points recursively ---
forecast = []
last_window = scaled_train[-window_size:].reshape(1, window_size, 1)
for _ in range(future_steps):
    pred = model.predict(last_window, verbose=0)
    forecast.append(pred[0, 0])
    last_window = np.append(last_window[:, 1:, :], [[[pred[0, 0]]]], axis=1)

forecast = np.array(forecast).reshape(-1, 1)
predicted_future = scaler.inverse_transform(forecast)

# --- Compare percentage changes ---
pred_change = ((predicted_future[-1] - predicted_future[0]) / predicted_future[0]) * 100
real_change = ((test_prices[-1] - test_prices[0]) / test_prices[0]) * 100

print("FORECAST RESULTS (Recursive 48h prediction)")
print(f"Predicted 48h Change: {pred_change[0]:.2f}%")
print(f"Actual 48h Change:    {real_change[0]:.2f}%")

# --- Plot results ---
plt.figure(figsize=(10, 5))
plt.plot(y_true, color="black", label="Actual (last 48h)")
plt.plot(pred_test, color="blue", label="Model Test Prediction (sliding)")
plt.plot(predicted_future, "--", color="green", label="Recursive Forecast (48h ahead)")
plt.title(f"{company} — Backtest vs Forecast")
plt.xlabel("Time Steps (hours)")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()
