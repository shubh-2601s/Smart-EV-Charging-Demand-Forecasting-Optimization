import pandas as pd

# Load datasets with timestamp parsing
ev = pd.read_csv("ev_usage.csv", parse_dates=["timestamp"])
weather = pd.read_csv("weather.csv", parse_dates=["timestamp"])
traffic = pd.read_csv("traffic.csv", parse_dates=["timestamp"])

# Merge EV usage with weather on timestamp + location
ev_weather = pd.merge(ev, weather, on=["timestamp", "location"])

# Merge traffic data into the combined dataset
full_df = pd.merge(ev_weather, traffic, on=["timestamp", "location"])

# Save the merged dataframe to a CSV file for future use
full_df.to_csv("full_merged_data.csv", index=False)

# Show the final merged dataframe shape and preview
print("Final merged dataset saved as 'full_merged_data.csv'")
print("Shape:", full_df.shape)
print(full_df.head())

