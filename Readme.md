# CO₂ Price Forecasting

This project predicts future European CO₂ certificate prices (EU ETS) using historical market and energy-related data combined with machine learning and classical time-series forecasting models.

## Overview

The goal of this project is to forecast EU Emissions Trading System (EU ETS) carbon allowance prices based on:

* historical CO₂ prices
* electricity generation mix
* electricity market prices
* carbon intensity of electricity generation
* coal prices
* natural gas prices
* EUR/USD exchange rate

Several forecasting approaches are implemented and compared:

* **LightGBM** for tabular machine learning regression
* **SARIMA / SARIMAX** for classical statistical time-series forecasting
* **LSTM** for deep learning sequence forecasting

---

## Data

The following data sources are used:

* CO₂ prices from Instrat
  https://energy.instrat.pl/en/prices/eu-ets/

* Electricity generation, carbon intensity, and electricity prices from Ember

* Coal prices, natural gas prices, and EUR/USD exchange rate from Federal Reserve Bank of St. Louis
  https://fred.stlouisfed.org/

Example FRED series:

* PNGASEUUSDM – Natural Gas Price Europe
* PCOALAUUSDM – Coal Price Australia
* DEXUSEU – USD/EUR Exchange Rate

---

## Features

The project includes extensive feature engineering:

* lag features (1, 3, 7, 30, 90 days)
* rolling mean / rolling standard deviation
* seasonal features:

  * month
  * weekday
  * sine/cosine cyclic encoding
* interpolation of missing values
* extrapolation of future feature scenarios:

  * linear trend
  * Monte Carlo simulation
  * Holt-Winters trend/seasonality extrapolation

---

## Models

### LightGBM

Gradient boosting model trained on engineered tabular features.

Advantages:

* handles nonlinear relationships
* fast training
* feature importance analysis

---

### SARIMA / SARIMAX

Classical statistical forecasting models.

Used for:

* univariate forecasting (SARIMA)
* multivariate forecasting with exogenous variables (SARIMAX)

---

### LSTM

Recurrent neural network for sequence forecasting.

Implemented in PyTorch.

---

## Installation

Clone the repository:

```bash id="4n5gfw"
git clone <your_repo_url>
cd <repo_name>
```

Install dependencies:

```bash id="d7h50v"
pip install -r requirements.txt
```

---

## Usage

### 1. Download raw data

Download all required datasets:

```bash id="jlwmgt"
python download_data_sources.py
```

This downloads:

* EU ETS CO₂ prices
* Electricity generation data
* Carbon intensity data
* Electricity prices
* Coal prices
* Natural gas prices
* EUR/USD exchange rate

---

### 2. Prepare and explore data

Merge datasets, clean missing values, and generate features:

```bash id="i5hkzs"
python explore.py
```

---

### 3. Train / evaluate forecasting models

#### LightGBM

```bash id="n7q9wn"
python lightgbm.py
```

#### SARIMA / SARIMAX

```bash id="g1j7mw"
python sarimax_forecast.py
```

#### LSTM

```bash id="37u4gx"
python lstm.py
```

---

## Evaluation Metrics

The following metrics are used to compare model performance:

* MSE (Mean Squared Error)
* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)
* R² Score

---

## Project Structure

```text id="l0x8j0"
.
├── download_data_sources.py   # Download raw data
├── explore.py                 # Data merge & feature engineering
├── lightgbm.py                # LightGBM forecasting
├── sarimax_forecast.py        # SARIMA / SARIMAX forecasting
├── lstm.py                    # LSTM forecasting
├── data/                      # Raw / processed data
├── plots/                     # Forecast visualizations
└── README.md
```

---

## Future Work

Possible improvements:

* hyperparameter tuning
* ensemble models (e.g. LightGBM + SARIMA)
* transformer-based forecasting
* improved uncertainty quantification

---

## Author

Jonas Fischer
PhD Mechanical Engineering
Ruhr University Bochum
