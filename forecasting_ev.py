import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

def prepare_data(df):
    # Aggregate energy usage hourly
    df_agg = df.groupby(pd.Grouper(key='timestamp', freq='h'))['energy_kwh'].sum().reset_index()
    df_agg.rename(columns={'timestamp': 'ds', 'energy_kwh': 'y'}, inplace=True)

    # Prepare regressors (mean values per hour)
    regressors_df = df.groupby(pd.Grouper(key='timestamp', freq='h')).agg({
        'temperature': 'mean',
        'rain_mm': 'mean',
        'traffic_norm': 'mean',
        'energy_kwh_lag1': 'mean',
        'energy_kwh_roll3': 'mean'
    }).reset_index()
    regressors_df.rename(columns={'timestamp': 'ds'}, inplace=True)

    # Merge target and regressors
    df_final = pd.merge(df_agg, regressors_df, on='ds')
    return df_final

def train_test_split(df, test_ratio=0.2):
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test

def train_prophet_model(train):
    model = Prophet()
    # Add regressors
    for regressor in ['temperature', 'rain_mm', 'traffic_norm', 'energy_kwh_lag1', 'energy_kwh_roll3']:
        model.add_regressor(regressor)
    model.fit(train)
    return model

def forecast_and_evaluate(model, train, test):
    # Create dataframe to forecast on test dates
    future = test[['ds', 'temperature', 'rain_mm', 'traffic_norm', 'energy_kwh_lag1', 'energy_kwh_roll3']]
    forecast = model.predict(future)

    # Evaluation metrics
    mae = mean_absolute_error(test['y'], forecast['yhat'])

    mse = mean_squared_error(test['y'], forecast['yhat'])
    rmse = mse ** 0.5

    return forecast, mae, rmse

def main():
    print("🔄 Loading featured dataset...")
    df = pd.read_csv("full_merged_data_featured.csv", parse_dates=["timestamp"])

    print("🔧 Preparing data for modeling...")
    df_final = prepare_data(df)

    print("🚦 Splitting data into train and test...")
    train_df, test_df = train_test_split(df_final)

    print("⚙️ Training Prophet model with regressors...")
    model = train_prophet_model(train_df)

    print("📈 Forecasting on test set and evaluating...")
    forecast, mae, rmse = forecast_and_evaluate(model, train_df, test_df)

    print(f"Model Evaluation:\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}")

    # Save forecast results
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv("forecast_results.csv", index=False)
    print("✅ Forecast results saved to 'forecast_results.csv'")

if __name__ == "__main__":
    main()
