import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

# Load the feature-engineered dataset
df = pd.read_csv("full_merged_data_featured.csv", parse_dates=['timestamp'])

# Forecasting output directory
os.makedirs("forecasts", exist_ok=True)

# Get unique station IDs
station_ids = df['station_id'].unique()

# Loop over each station to create a separate model
for station in station_ids:
    print(f"🔮 Forecasting for Station ID: {station}")

    # Filter for this station
    station_df = df[df['station_id'] == station][['timestamp', 'energy_kwh']].rename(
        columns={'timestamp': 'ds', 'energy_kwh': 'y'}
    )

    # Resample hourly to handle any missing timestamps
    station_df = station_df.set_index('ds').resample('H').mean().fillna(0).reset_index()

    # Initialize and fit Prophet
    model = Prophet()
    model.fit(station_df)

    # Create future dataframe for next 7 days
    future = model.make_future_dataframe(periods=168, freq='H')  # 7 days × 24 hrs

    # Predict
    forecast = model.predict(future)

    # Save forecast
    forecast_file = f"forecasts/forecast_station_{station}.csv"
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(forecast_file, index=False)
    print(f"✅ Forecast saved: {forecast_file}")

    # Optional: Plot forecast
    fig = model.plot(forecast)
    plt.title(f"EV Charging Demand Forecast - Station {station}")
    plt.xlabel("Date")
    plt.ylabel("Predicted Energy Usage (kWh)")
    plt.tight_layout()
    plt.savefig(f"forecasts/forecast_station_{station}.png")
    plt.close()

print("📈 All forecasts complete.")
