import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AutoReg

# Load soybean data
df = pd.read_csv("soybeans.csv")
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)
df = df.asfreq("MS")

# 1. Nominal vs Real Price
plt.plot(df.index, df["price"], label="Nominal")
plt.plot(df.index, df["price_real"], label="Real")
plt.title("Nominal vs Real Soybean Prices")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("lab15_nominal_real.png")
plt.clf()

# 2. Seasonal decomposition
decomp = seasonal_decompose(df["price"], model="additive", period=12)
seasonal = decomp.seasonal
print("Monthly Seasonal Effects:")
print(seasonal.groupby(seasonal.index.month).mean())

# 3. Simple Moving Averages
df["SMA_3"] = df["price"].rolling(3).mean()
df["SMA_12"] = df["price"].rolling(12).mean()
df["SMA_48"] = df["price"].rolling(48).mean()
plt.plot(df.index, df["price"], alpha=0.3)
plt.plot(df.index, df["SMA_3"], label="3-Month")
plt.plot(df.index, df["SMA_12"], label="12-Month")
plt.plot(df.index, df["SMA_48"], label="48-Month")
plt.title("Simple Moving Averages")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("lab15_sma.png")
plt.clf()

# 4. Log-differencing
df["log_price"] = np.log(df["price"])
df["log_diff"] = df["log_price"].diff()
df["log_diff"].plot(title="Log Differenced Prices")
plt.tight_layout()
plt.grid(True)
plt.savefig("lab15_logdiff.png")
plt.clf()

# 5. Auto-Regressive model
log_diff = df["log_diff"].dropna()
ar_model = AutoReg(log_diff, lags=6).fit()
print("AR(6) Coefficients:")
print(ar_model.params)

# 6. Forecast next 6 months
forecast = ar_model.predict(start=len(log_diff), end=len(log_diff)+5)
recent = log_diff[-36:]
forecast_idx = pd.date_range(start=log_diff.index[-1] + pd.DateOffset(months=1), periods=6, freq="MS")
combined = pd.concat([recent, pd.Series(forecast, index=forecast_idx)])
plt.plot(combined.index, combined.values)
plt.axvline(log_diff.index[-1], color="red", linestyle="--")
plt.title("Forecast: Aug 2020 – Jan 2021")
plt.grid(True)
plt.tight_layout()
plt.savefig("lab15_forecast.png")