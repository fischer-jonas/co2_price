# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 18:33:18 2025

@author: fisch
"""
import explore

from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


df_merged = explore.explore()
df_fe =  explore.feature_engineering(df_merged)
X_train_scaled, X_test_scaled, y_train, y_test, scaler, df_extended = explore.split(df_fe)


model = LGBMRegressor(
    n_estimators=5000,
    learning_rate=0.01
)

model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    eval_metric="mse",
)

y_pred = model.predict(X_test_scaled)

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("MSE :", mse)
print("RMSE:", rmse)
print("MAE :", mae)
print("R²  :", r2)

plt.figure(figsize=(8, 5))
plt.plot(y_test,label="test")
plt.plot(y_pred,label="prediction)")
plt.xlabel("Days")
plt.ylabel("Price [€]")
plt.title("Test vs. Prediction")
plt.legend()
plt.show()


#%%%%%%%%%%%%%%%%%%%
importance_gain = model.booster_.feature_importance(importance_type='gain')
importance_split = model.booster_.feature_importance(importance_type='split')
importance_df = pd.DataFrame({
    'feature': df_extended.drop(columns=['price', 'volume']).columns,
    'importance_gain': importance_gain,
    'importance_split': importance_split
})
importance_df = importance_df.sort_values(by='importance_gain', ascending=False)
print(importance_df)

plt.figure(figsize=(10,6))
plt.barh(importance_df['feature'][0:20], importance_df['importance_gain'][0:20])
plt.gca().invert_yaxis()  # höchstes oben
plt.xlabel("Feature Importance [Log]")
plt.ylabel("Features")
plt.title("LightGBM Feature Importance")
plt.xscale('log')
plt.show()

#%%%%%%%%%%%%

#Predict future

df = df_merged.copy()

feature_cols = df.columns


days_to_predict = 120
lookback_days = 365  # days for the trend


# DataFrame for the future
future_df = pd.DataFrame(index=range(len(df), len(df) + days_to_predict))

for col in feature_cols:
    last_values = df[col].tail(lookback_days)
    slope = (last_values.iloc[-1] - last_values.iloc[0]) / (lookback_days - 1)
    future_df[col] = [last_values.iloc[-1] + slope * i for i in range(1, days_to_predict + 1)]


if isinstance(df.index, pd.DatetimeIndex):
    future_df.index = pd.date_range(df.index[-1] + pd.Timedelta(days=1),
                                    periods=days_to_predict, freq='D')

print(future_df.head())
linear_price = future_df["price"]

df_tail = df.tail(90)
future_df_combined = pd.concat([df_tail, future_df])
future_df_extended = explore.feature_engineering(future_df_combined)
X_future = future_df_extended.drop(columns=['price', 'volume']).to_numpy()

X_future_scaled = scaler.transform(X_future)
future_predictions  = model.predict(X_future_scaled)

plt.figure(figsize=(14, 5))
plt.plot(df["price"].iloc[-180:], label="Historisch")
plt.plot(future_df.index.to_numpy(),future_predictions , label="Forecast")
plt.plot(linear_price.index.to_numpy(),linear_price.to_numpy() , label="Linear Forecast")
plt.legend()
plt.title(f"LightGBM Forecast – Nächste {days_to_predict} Tage")
plt.show()

#%%%%%%%%%%%%%%%%

# Monte-Carlo Simulation


monte_carlo_runs = 1000   
mc_horizon = days_to_predict

future_mc_p10 = pd.DataFrame(index=future_df.index)
future_mc = pd.DataFrame(index=future_df.index)
future_mc_p90 = pd.DataFrame(index=future_df.index)


for col in feature_cols:
    last_values = df[col].tail(lookback_days).astype(float)

    # Trend (linear)
    slope = (last_values.iloc[-1] - last_values.iloc[0]) / (lookback_days - 1)

    # Volatility
    sigma = last_values.diff().std()
    if sigma < 1e-8:
        sigma = 0.001  
    print(f"Sigma {col} = {sigma}")

    # Monte-Carlo runs
    sims = []
    last = last_values.iloc[-1]
    for run in range(monte_carlo_runs):
        values = []
        curr = last
        for t in range(mc_horizon):
            noise = np.random.normal(0, sigma)
            curr = curr + slope + noise
            values.append(curr)
        sims.append(values)

    sims = np.array(sims)

    future_mc_p10[col] = np.percentile(sims, 10, axis=0)
    future_mc[col] = sims.mean(axis=0)
    future_mc_p90[col] = np.percentile(sims, 90, axis=0)
    
    bands=[future_mc_p10,future_mc,future_mc_p90]
    band_labels=["10 % Quartile","50 % Quartile","90 % Quartile"]
    
plt.figure(figsize=(14, 5))  
i=0
for b in bands:
    df_tail = df.tail(90)
    future_df_combined_mc = pd.concat([df_tail, b])
    future_df_extended_mc = explore.feature_engineering(future_df_combined_mc)
    X_future_mc = future_df_extended_mc.drop(columns=['price', 'volume']).to_numpy()
    
    X_future_mc_scaled = scaler.transform(X_future_mc)
    future_predictions_mc  = model.predict(X_future_mc_scaled)
    plt.plot(future_df_extended_mc.index.to_numpy(),future_predictions_mc , label=band_labels[i])
    i=+1
    
plt.plot(df["price"].iloc[-180:], label="History")        
plt.plot(linear_price.index.to_numpy(),linear_price.to_numpy() , label="Linear Forecast")
plt.legend()
plt.title(f"LightGBM Forecast – Next {days_to_predict} Days")
plt.show()
