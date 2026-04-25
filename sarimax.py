import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import explore  


# ================================================================
# 1) Daten laden
# ================================================================
df = explore.explore()            # dein zentraler Daten-Loader
df = df.copy()

# Index sicherstellen
if not isinstance(df.index, pd.DatetimeIndex):
    df.index = pd.to_datetime(df.index)

df = df.sort_index()


# ================================================================
# 2) Zielvariable & Feature-Auswahl
# ================================================================
target_col = "price"

# Alle Features außer price
feature_cols = [c for c in df.columns if c != target_col]

y = df[target_col]
X = df[feature_cols]

print("Target:", target_col)
print("Features:", len(feature_cols))


# ================================================================
# 3) Zeitgerechter Train/Test-Split
# ================================================================
train_ratio = 0.8
split_idx = int(len(df) * train_ratio)

y_train = y.iloc[:split_idx]
y_test  = y.iloc[split_idx:]

X_train = X.iloc[:split_idx]
X_test  = X.iloc[split_idx:]

print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")


# ================================================================
# 4) SARIMAX Modell trainieren
# ================================================================
model = sm.tsa.statespace.SARIMAX(
    endog=y_train,
    exog=X_train,
    order=(1, 1, 1),                 # ARIMA-Teil
    seasonal_order=(0,0,0,0),#(1, 1, 1, 120),   # jährliche Saisonalität
    enforce_stationarity=False,
    enforce_invertibility=False
)

result = model.fit(disp=False)
print(result.summary())


# ================================================================
# 5) Test-Forecast
# ================================================================
fc_test = result.get_forecast(steps=len(y_test), exog=X_test)
pred_mean = fc_test.predicted_mean
pred_ci   = fc_test.conf_int()


mse  = mean_squared_error(y_test, pred_mean)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, pred_mean)
r2   = r2_score(y_test, pred_mean)

print("MSE:  ", mse)
print("RMSE: ", rmse)
print("MAE:  ", mae)
print("R²:   ", r2)



# ================================================================
# 6) Plot: Test-Forecast
# ================================================================
plt.figure(figsize=(14,5))
#plt.plot(y_train, label="Train")
plt.plot(y_test.index.to_numpy(),y_test.to_numpy(), label="Test (Real)")
plt.plot(y_test.index.to_numpy(),pred_mean.to_numpy(), label="SARIMAX Forecast (Test)")

# plt.fill_between(pred_ci.index,
#                  pred_ci.iloc[:,0],
#                  pred_ci.iloc[:,1],
#                  alpha=0.2, color="orange")

plt.legend()
plt.title("SARIMAX – Test Forecast")
plt.show()


# ================================================================
# 7) Zukunftsfeatures generieren (Linearer Trend pro Feature)
# ================================================================
days_to_predict = 120
lookback_days = 365

future_index = pd.date_range(
    start=df.index[-1] + pd.Timedelta(days=1),
    periods=days_to_predict,
    freq="D"
)

future_df = pd.DataFrame(index=future_index)

for col in feature_cols:
    last = df[col].tail(lookback_days).astype(float)
    slope = (last.iloc[-1] - last.iloc[0]) / (lookback_days - 1)

    future_df[col] = [last.iloc[-1] + slope * i for i in range(1, days_to_predict + 1)]


# ================================================================
# 8) Zukunfts-Forecast mit SARIMAX
# ================================================================
fc_future = result.get_forecast(steps=days_to_predict, exog=future_df)

future_mean = fc_future.predicted_mean
future_ci   = fc_future.conf_int()


# ================================================================
# 9) Plot: Zukunfts-Forecast
# ================================================================
plt.figure(figsize=(14,5))

plt.plot(df[target_col].iloc[-200:], label="Historisch (letzte 200 Tage)")
plt.plot(future_index.to_numpy(),future_mean.to_numpy(), label="SARIMAX Zukunfts-Forecast", color="blue")

# plt.fill_between(future_ci.index,
#                  future_ci.iloc[:,0],
#                  future_ci.iloc[:,1],
#                  alpha=0.2, color="blue")

plt.legend()
plt.title(f"SARIMAX Forecast – Nächste {days_to_predict} Tage")
plt.show()
