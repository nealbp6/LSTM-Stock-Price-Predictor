import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score

def plot_results(y_true, y_pred, title="True vs Predicted", xlabel="Samples", ylabel="Values"):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='True Values', color='b')
    plt.plot(y_pred, label='Predicted Values', color='r', linestyle='--')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2