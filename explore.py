import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates


pd.set_option('display.max_columns', None)

def explore():
    #%%%%%%%%%%%%%
    data_co2 = pd.read_csv("prices_eu_ets_all.csv")
    data_co2["date"] = pd.to_datetime(data_co2['date'], format='%d.%m.%Y')
    data_co2 = data_co2.rename(columns={'indeks': 'index'})
    data_co2["index"]=data_co2["index"].str.extract(r'(\d+)').astype(int)
    
    print(data_co2.head())
    
    plt.plot(data_co2["date"].to_numpy(),data_co2["price"].to_numpy())
    ax = plt.gca()
    ax.set_xticks(data_co2["date"][::100]) # Show only every 100th date on x-axis
    plt.xticks(rotation=45)
    plt.title("CO2 Price evolution")
    plt.xlabel("Date")
    plt.ylabel("Price [EUR/t]")
    plt.tight_layout()
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=12))  # every 3 months (adjust)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.show()
    
    
    plt.bar(data_co2["date"],data_co2["volume"])
    plt.ylabel("Volumes [t]")
    plt.show()
    
    #%%%%%%%%%%%%%
    data_carbon_intensity = pd.read_csv("ember_monthly_carbon-intensity - All electricity sources - EU.csv")
    data_carbon_intensity["date"] = pd.to_datetime(data_carbon_intensity['date'], format='%Y-%m')
    data_carbon_intensity.drop(columns=["entity","entity_code",'is_aggregate_entity','emissions_intensity_yoy_change_gco2_per_kwh','emissions_intensity_yoy_change_pct'],inplace=True)
    print(data_carbon_intensity.head())
    
    plt.plot(data_carbon_intensity["date"].to_numpy(),data_carbon_intensity["emissions_intensity_gco2_per_kwh"].to_numpy())
    ax = plt.gca()
    plt.title("Carbon-intensity")
    plt.xlabel("Date")
    plt.ylabel("Carbon intensity [gCO2/KWh]")
    plt.tight_layout()
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=12))  # every 3 months (adjust)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.show()
    
    #%%%%%%%%%%%%%
    data_elec_prices = pd.read_csv("european_wholesale_electricity_price_data_monthly.csv")
    data_elec_prices["date"] = pd.to_datetime(data_elec_prices['Date'], format='%Y-%m-%d')
    data_elec_prices.drop(columns=["Date","ISO3 Code"],inplace=True)
    print(data_elec_prices.head())
    
    for c in data_elec_prices["Country"].unique():
        country_data = data_elec_prices[data_elec_prices["Country"]==c]
        plt.plot(country_data["date"].to_numpy(),country_data["Price (EUR/MWhe)"].to_numpy(),label=c)
    ax = plt.gca()
    plt.title("Electricity price")
    plt.xlabel("Date")
    plt.ylabel("Electricity price [EUR/MWhe]")
    plt.tight_layout()
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=12))  # every 3 months (adjust)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.legend(ncol=4,loc='upper center', bbox_to_anchor=(0.5, -0.15))
    plt.show()
    
    data_mean_elec_prices = (
        data_elec_prices
        .groupby("date")["Price (EUR/MWhe)"]
        .agg(["mean", "std"])
        .reset_index()
    )
    
    data_mean_elec_prices = data_mean_elec_prices.rename(columns={
    "mean": "elec_price_mean",
    "std": "elec_price_std"
    })
    
    plt.plot(data_mean_elec_prices["date"].to_numpy(),data_mean_elec_prices["elec_price_mean"].to_numpy(),label="mean")
    plt.plot(data_mean_elec_prices["date"].to_numpy(),data_mean_elec_prices["elec_price_mean"].to_numpy()+data_mean_elec_prices["elec_price_std"].to_numpy(),linestyle=":",color="grey",label="std")
    plt.plot(data_mean_elec_prices["date"].to_numpy(),data_mean_elec_prices["elec_price_mean"].to_numpy()-data_mean_elec_prices["elec_price_std"].to_numpy(),linestyle=":",color="grey")
    plt.legend(ncol=4,loc='upper center', bbox_to_anchor=(0.5, -0.15))
    plt.show()
    
    #%%%%%%%%%%%%%
    data_elec_gen = pd.read_csv("ember_monthly_electricity-generation - All electricity sources - EU - breakdown.csv")
    data_elec_gen["date"] = pd.to_datetime(data_elec_gen['date'], format='%Y-%m')
    data_elec_gen.drop(columns=['entity', 'entity_code','is_aggregate_series', 'is_aggregate_entity','generation_kwh_per_capita', 'generation_yoy_change_twh',
                                   'generation_yoy_change_pct', 'generation_share_yoy_change_pct_points',
                                   'generation_yoy_change_kwh_per_capita', 'generation_ytd_twh',
                                   'generation_ytd_share_pct', 'generation_ytd_kwh_per_capita','generation_share_pct'],inplace=True)
    
    data_elec_gen = data_elec_gen.pivot_table(
        index="date",
        columns="series",
        values="generation_twh"
    ).reset_index()
    print(data_elec_gen.head())
    
    
    plt.scatter(data_elec_gen["date"].dt.month.to_numpy(),data_elec_gen["Total generation"].to_numpy())
    plt.title("Total generation")
    plt.xlabel("Month")
    plt.ylabel("Total generation [TWh]")
    plt.show()
    
    for c in data_elec_gen.keys():
        if not c=="date" and not c=="Total generation":
            plt.plot(data_elec_gen["date"].to_numpy(),data_elec_gen[c].to_numpy(),label=c)
    plt.title("Generation by methods")
    plt.xlabel("Date")
    plt.ylabel("Generation [TWh]")
    plt.legend(ncol=4,loc='upper center', bbox_to_anchor=(0.5, -0.15))
    plt.show()
    
    #%%%%%%%%%%%%%
    data_dollar_euro = pd.read_csv("DEXUSEU.csv") #Euro per dollar
    data_dollar_euro["date"] = pd.to_datetime(data_dollar_euro['observation_date'], format='%Y-%m-%d')
    data_dollar_euro.drop(columns=["observation_date"],inplace=True)
    data_dollar_euro = data_dollar_euro[data_dollar_euro["date"]>="2012-01-03"]
    data_dollar_euro.reset_index()
    
    data_gas_prices = pd.read_csv("PNGASEUUSDM.csv") #U.S. Dollars per Million Metric British Thermal Unit,
    data_gas_prices["date"] = pd.to_datetime(data_gas_prices['observation_date'], format='%Y-%m-%d')
    data_gas_prices.drop(columns=["observation_date"],inplace=True)
    data_gas_prices = data_gas_prices[data_gas_prices["date"]>="2012-01-03"]
    data_gas_prices.reset_index()
    
    data_coal_prices = pd.read_csv("PCOALAUUSDM.csv") #$ per metric ton
    data_coal_prices["date"] = pd.to_datetime(data_coal_prices['observation_date'], format='%Y-%m-%d')
    data_coal_prices.drop(columns=["observation_date"],inplace=True)
    data_coal_prices = data_coal_prices[data_coal_prices["date"]>="2012-01-03"]
    data_gas_prices.reset_index()
    
    fuel_prices = data_gas_prices.merge(data_dollar_euro, on="date", how="outer") \
                   .merge(data_coal_prices, on="date", how="outer")
    fuel_prices = fuel_prices.dropna(subset=["PNGASEUUSDM", "PCOALAUUSDM"], how="all")
    fuel_prices = fuel_prices.sort_values("date")
    fuel_prices = fuel_prices.interpolate()
    
    fuel_prices["Gas price euro"] = fuel_prices["PNGASEUUSDM"]/fuel_prices["DEXUSEU"]/0.2931 #MWh instead of MMBTU
    fuel_prices["Coal price euro"] = fuel_prices["PCOALAUUSDM"]/fuel_prices["DEXUSEU"]/6.945 #MWh instead of metric ton
    fuel_prices.drop(columns=["PNGASEUUSDM","PCOALAUUSDM"],inplace=True)
    fuel_prices = fuel_prices.rename(columns={"DEXUSEU": "Euro per dollar"})
    print(fuel_prices.head())
    
    plt.plot(fuel_prices["date"].to_numpy(),fuel_prices["Gas price euro"].to_numpy(),label="Gas price euro")
    plt.plot(fuel_prices["date"].to_numpy(),fuel_prices["Coal price euro"].to_numpy(),label="Coal price euro")
    plt.legend(ncol=2,loc='upper center', bbox_to_anchor=(0.5, -0.15))
    plt.xlabel("Date")
    plt.ylabel("Price [€/MWH]")
    plt.show()
    
    #%%%%%%%%%%%%%%%%%%%%
    #Join all DFs
    dfs = [data_co2, data_carbon_intensity, data_mean_elec_prices, data_elec_gen,fuel_prices]
    dfs_indexed = [df.set_index("date") for df in dfs]
    df_merged = dfs_indexed[0].join(dfs_indexed[1:], how="outer").reset_index()
    df_merged = df_merged.set_index("date").interpolate(method="time").reset_index()
    
    #Trim to earliest point in time with all data
    #TODO: check if extrapolation may be better
    df_merged = df_merged.dropna()
    print(df_merged.info())
    
    
    #Lags
    df_merged = df_merged.sort_values("date").reset_index(drop=True)
    df_merged.set_index("date", inplace=True)
    df_merged = df_merged.copy()
    return df_merged
    
def feature_engineering(df_merged):
    
    lags = [1, 3, 7, 30, 90]  # days
    dfs_lags = []
    for lag in lags:
        lagged = df_merged.shift(lag).add_suffix(f"_lag{lag}")
        dfs_lags.append(lagged)
    #df_merged = pd.concat([df_merged] + dfs_lags, axis=1)
            
    
    #Rolling window
    windows = [30,60]  #days
    
    dfs_to_concat = []
    for w in windows:
        roll_mean = df_merged.rolling(w).mean().add_suffix(f"_roll_mean{w}")
        # roll_std  = df_merged.rolling(w).std().add_suffix(f"_roll_std{w}")
        # dfs_to_concat.extend([roll_mean, roll_std])
        dfs_to_concat.extend([roll_mean])
    df_merged = pd.concat([df_merged] +dfs_lags+ dfs_to_concat, axis=1)
            
    df_merged = df_merged.copy() #Defrag
    df_merged = df_merged.dropna()
    #Repating Months and days
    df_merged["month"] = df_merged.index.month.astype(float)
    # Sin/Cos Transformation
    df_merged["month_sin"] = np.sin(2 * np.pi * df_merged["month"] / 12)
    df_merged["month_cos"] = np.cos(2 * np.pi * df_merged["month"] / 12)
    
    df_merged["day_of_week"] = df_merged.index.dayofweek.astype(float)  # 0=Monday, 6=Sunday
    df_merged["dow_sin"] = np.sin(2 * np.pi * df_merged["day_of_week"] / 7)
    df_merged["dow_cos"] = np.cos(2 * np.pi * df_merged["day_of_week"] / 7)
    
    print(df_merged.info())
    
    return df_merged
    
    #%%%%%%%%%%%%%
def split(df_merged):
    X = df_merged.drop(columns=['price', 'volume']).to_numpy()
    y = df_merged["price"].to_numpy()
    
    
    train_size = int(len(X) * 0.8)  # 80% Training, 20% Test
    
    X_train = X[:train_size]
    X_test  = X[train_size:]
    y_train = y[:train_size]
    y_test  = y[train_size:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    df_extended = df_merged.copy()

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, df_extended

if __name__ == "__main__":
    df_merged = explore()
    df_merged = feature_engineering(df_merged)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, df_extended = split(df_merged)