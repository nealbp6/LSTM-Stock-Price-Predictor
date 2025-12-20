# LSTM Stock Price Predictor

## Overview

This repository is a complete PyTorch-first rewrite of the LSTM forecasting pipeline. The code is modular: data download, preprocessing/feature engineering, dataset/window creation, PyTorch model definitions and training loops, plus backtesting/forecasting utilities.

## Why PyTorch?

- Explicit training loops for full control (custom losses, schedulers, mixed precision).
- Easier debugging and introspection of tensors and gradients.
- Faster iteration for custom research experiments and flexible model design.


## Project layout (short)

- `fetch_data.py`  — download market data (yfinance) and save a local cache.
- `prepare_inputs.py` — feature engineering, scaling, and windowed dataset creation.
- `features.py`     — technical indicators and derived feature helpers.
- `model.py`        — PyTorch model definitions and training/evaluation entrypoints.
- `result.py`       — backtesting, forecasting, and metric reporting.
- `old/`            — legacy TensorFlow/Keras code (kept only for reference).


## Quickstart (PyTorch only):

1) Create and activate a virtual environment (Windows):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies:

```powershell
pip install -r requirements.txt
```

3) Run:

```powershell
python model.py
```

## What to edit for common changes

- Change asset/timeframe: modify `fetch_data.py`.
- Adjust model lookback: set `WINDOW_SIZE` (usually in `prepare_inputs.py`).
- Change features: edit `features.py` or the features list in `prepare_inputs.py`.
- Training settings: edit `model.py` (optimizer, lr, batch, epochs, device).

## Documentation

All project documentation, including detailed explanations of functions, model architecture, and training procedure, can be found in the pdf document named `Documentaion.pdf`.

## AI Disclosure

During the development of this stock prediction project, AI-powered tools were used to assist in structuring, explaining, and documenting the code. Specifically:  

- **ChatGPT and other models:** Assisted in generating explanations, improving readability, and drafting documentation.  

> ⚠️ Important: All logic, model architecture, and implementation were designed, reviewed, and verified by the human developer. AI tools were used only to assist productivity.

---

## Credits

- **Developer:** Neal  
- **AI Tools Used:** ChatGPT  
- **Open-Source Libraries / Frameworks:**  
  - Python  
  - NumPy  
  - PyTorch  
  - scikit-learn  
  - Matplotlib  
  - yfinance  
  - TA-Lib  
- **APIs Used:** yfinance API  
- **Inspiration / References:** https://www.youtube.com/watch?v=IJ50ew8wi-0&t; https://www.youtube.com/watch?v=b61DPVFX03I&; https://chatgpt.com


## License

This project is licensed under the MIT License.

> © **2025 Neal**  

> *Feel free to modify, improve, and share — just include credit to the original author.*