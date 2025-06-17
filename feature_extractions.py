import pandas as pd
import numpy as np

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Summer'
    elif month in [6, 7, 8]:
        return 'Monsoon'
    else:
        return 'Autumn'

def part_of_day(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

def traffic_level(volume):
    if volume < 400:
        return 'Low'
    elif volume < 600:
        return 'Medium'
    else:
        return 'High'

def main():
    print("🔄 Loading merged dataset...")
    full_df = pd.read_csv("full_merged_data.csv", parse_dates=["timestamp"])

    print("🔧 Performing feature engineering...")

    # --- Time Features ---
    full_df['hour'] = full_df['timestamp'].dt.hour
    full_df['day_of_week'] = full_df['timestamp'].dt.day_name()
    full_df['is_weekend'] = full_df['day_of_week'].isin(['Saturday', 'Sunday'])
    full_df['month'] = full_df['timestamp'].dt.month
    full_df['quarter'] = full_df['timestamp'].dt.quarter
    full_df['day_of_year'] = full_df['timestamp'].dt.dayofyear
    full_df['week_of_year'] = full_df['timestamp'].dt.isocalendar().week
    full_df['season'] = full_df['month'].apply(get_season)
    full_df['part_of_day'] = full_df['hour'].apply(part_of_day)

    # --- Weather Features ---
    full_df['rain_flag'] = (full_df['rain_mm'] > 0).astype(int)
    full_df = full_df.sort_values(['station_id', 'timestamp'])
    full_df['temp_diff'] = full_df.groupby('station_id')['temperature'].diff().fillna(0)

    # --- Traffic Features ---
    full_df['traffic_level'] = full_df['traffic_volume'].apply(traffic_level)
    full_df['traffic_level'] = full_df['traffic_level'].astype('category')
    full_df['traffic_norm'] = (full_df['traffic_volume'] - full_df['traffic_volume'].mean()) / full_df['traffic_volume'].std()

    # --- EV Usage Features ---
    full_df['energy_kwh_lag1'] = full_df.groupby('station_id')['energy_kwh'].shift(1).fillna(0)
    full_df['energy_kwh_roll3'] = (
        full_df.groupby('station_id')['energy_kwh']
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(0, drop=True)
    )

    # --- Convert to categories ---
    cat_cols = ['day_of_week', 'season', 'part_of_day']
    for col in cat_cols:
        full_df[col] = full_df[col].astype('category')

    # --- Handle missing values (Only numeric columns) ---
    num_cols = full_df.select_dtypes(include=[np.number]).columns
    full_df[num_cols] = full_df[num_cols].fillna(0)

    # --- Save feature engineered dataset ---
    output_file = "full_merged_data_featured.csv"
    full_df.to_csv(output_file, index=False)
    print(f"✅ Feature-engineered dataset saved as '{output_file}'")

    # Preview
    print("\n🔍 Sample preview:")
    print(full_df.head(10))

if __name__ == "__main__":
    main()
