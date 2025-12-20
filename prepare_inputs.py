import numpy as np
import talib

def regression_data_preparation(WINDOW, MIN_SNR, RETURNS):
    features = []
    removed_indices = []

    # determine average trend and volatility over rolling windows
    total_vol = 0.0
    count = 0
    for i in range(WINDOW, len(RETURNS)):
        window_returns = RETURNS[i - WINDOW : i].astype(float)
        vol = float(np.std(window_returns))
        total_vol += vol
        count += 1
    average_volatility = total_vol / count if count > 0 else 0        

    for i in range(WINDOW, len(RETURNS)):
        y = RETURNS[i - WINDOW : i].astype(float) # Extract window of returns
        x = np.arange(WINDOW, dtype=float) # Time indices for regression
 
        # Weighted linear regression
        weights = np.linspace(1.0, float(WINDOW), WINDOW)

        x_w_mean = np.average(x, weights=weights)
        y_w_mean = np.average(y, weights=weights)

        numerator = np.sum(weights * (x - x_w_mean) * (y - y_w_mean))
        denominator = np.sum(weights * (x - x_w_mean) ** 2)

        if denominator == 0:
            removed_indices.append(i)
            continue

        slope = float(numerator / denominator)

        # Signal-to-noise ratio
        snr = abs(slope) / average_volatility

        # Filter: discard very weak signals
        if snr < MIN_SNR:
            removed_indices.append(i)
            continue

        # Feature set
        features.append([
            slope,              # trend
            slope / average_volatility,  # normalized trend
            snr                 # signal-to-noise ratio
        ])

    features = np.array(features, dtype=float)
    removed_indices = np.array(removed_indices, dtype=int)

    print(f"Regression samples kept: {len(features)}")
    print(f"Regression samples removed: {len(removed_indices)}")

    return features, removed_indices

def returns_data_preparation(WINDOW, RETURNS, removed_idx):
    inputs_list = []
    labels_list = []

    for i in range(WINDOW, len(RETURNS)):
        if i in removed_idx:
            continue

        inputs = RETURNS[i - WINDOW : i].astype(float)  # input window
        label = RETURNS[i].astype(float)                # next return
        inputs_list.append(inputs)
        labels_list.append(label)

    return np.array(inputs_list, dtype=float), np.array(labels_list, dtype=float)


def technical_data_preparation(WINDOW, DATA, RETURNS, removed_idx): # window musst be at least 20
    technical_features = []

    for i in range(WINDOW, len(RETURNS)):
        if i in removed_idx:
            continue

        window_data = DATA[i - WINDOW : i]
        
        # Convert Series to numpy array
        close_prices = np.asarray(window_data["Close"], dtype=np.float64).ravel()

        # Example technical indicators
        sma10 = talib.SMA(close_prices, timeperiod=10)[-1]
        sma20 = talib.SMA(close_prices, timeperiod=20)[-1]
        ema20 = talib.EMA(close_prices, timeperiod=20)[-1]
        rsi14 = talib.RSI(close_prices, timeperiod=14)[-1]
        technical_features.append([sma10, sma20, ema20, rsi14])

    technical_features = np.array(technical_features, dtype=float)

    return technical_features
