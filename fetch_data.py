# fetch_data.py
import yfinance as yf
import numpy as np
import pandas as pd

def get_data(symbol, interval, period="max"):
    data = yf.download(
        symbol,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True
    )

    data = data.dropna()

    # Log returns (better for ML)
    returns = np.log(data["Close"]).diff().dropna()

    return returns, data
