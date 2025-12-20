import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

from fetch_data import get_data
from prepare_inputs import regression_data_preparation, returns_data_preparation, technical_data_preparation


def load_data(WINDOW, MIN_SNR): # MIN_SNR = 0.01
    returns, data = get_data("BTC-USD", "1h")
    returns = returns.values.flatten()
    
    features, removed_idx = regression_data_preparation(WINDOW, MIN_SNR, returns)
    inputs, labels = returns_data_preparation(WINDOW, returns, removed_idx)
    technical_features = technical_data_preparation(WINDOW, data, returns, removed_idx)

    x = np.hstack((features, technical_features, inputs))
    y = labels

    return x, y

def label_scale_data(labels):
    y_scaler = RobustScaler()

    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    labels_scaled = y_scaler.fit_transform(labels)

    return labels_scaled, y_scaler

def input_scale_data(features, technical_features, inputs):
    # Automatically handle 1D inputs 
    x_scaler_list = []
    regression_scaler = StandardScaler()
    returns_scaler = RobustScaler()
    technical_scaler = MinMaxScaler()

    if inputs.ndim == 1:
        inputs = inputs.reshape(-1, 1)

    features_scaled = regression_scaler.fit_transform(features)
    technical_features_scaled = technical_scaler.fit_transform(technical_features)
    inputs_scaled = returns_scaler.fit_transform(inputs)

    x_scaler_list.append(regression_scaler)
    x_scaler_list.append(technical_scaler) 
    x_scaler_list.append(returns_scaler) 

    input_scaled = np.hstack((features_scaled, technical_features_scaled, inputs_scaled))

    return input_scaled, x_scaler_list

def prepare_dataset(WINDOW, MIN_SNR, test_size):
    returns, data = get_data("BTC-USD", "1h")
    returns = returns.values.flatten()
    
    features, removed_idx = regression_data_preparation(WINDOW, MIN_SNR, returns)
    inputs, labels = returns_data_preparation(WINDOW, returns, removed_idx)
    technical_features = technical_data_preparation(WINDOW, data, returns, removed_idx)

    x, x_scaler_list = input_scale_data(features, technical_features, inputs)
    y, y_scaler = label_scale_data(labels)

    split_index = int(len(x) * (1 - test_size))
    x_train, x_test = x[:split_index], x[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return x_train, y_train, x_test, y_test, x_scaler_list, y_scaler
