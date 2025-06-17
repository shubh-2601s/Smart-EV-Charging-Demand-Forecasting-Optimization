import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directory
os.makedirs("eda_outputs", exist_ok=True)

# Load data
df = pd.read_csv("full_merged_data_featured.csv", parse_dates=['timestamp'])

# Save data summary
df.describe(include='all').to_csv("eda_outputs/summary_statistics.csv")
df.info(buf=open("eda_outputs/data_info.txt", "w"))

# Null value heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Value Heatmap")
plt.savefig("eda_outputs/missing_values_heatmap.png")
plt.close()

# Correlation heatmap (numerical only)
plt.figure(figsize=(14, 10))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_outputs/correlation_heatmap.png")
plt.close()

# Energy usage over time
plt.figure(figsize=(14, 6))
df.groupby("timestamp")["energy_kwh"].sum().plot()
plt.title("Total Energy Usage Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Energy (kWh)")
plt.tight_layout()
plt.savefig("eda_outputs/energy_over_time.png")
plt.close()

# Energy usage by hour
plt.figure(figsize=(10, 6))
sns.boxplot(x='hour', y='energy_kwh', data=df)
plt.title("Energy Usage by Hour of Day")
plt.savefig("eda_outputs/energy_by_hour.png")
plt.close()

# Energy usage by day of week
plt.figure(figsize=(10, 6))
sns.boxplot(x='day_of_week', y='energy_kwh', data=df, order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
plt.title("Energy Usage by Day of Week")
plt.savefig("eda_outputs/energy_by_dayofweek.png")
plt.close()

# Energy usage by traffic level
plt.figure(figsize=(8, 6))
sns.boxplot(x='traffic_level', y='energy_kwh', data=df)
plt.title("Energy Usage by Traffic Level")
plt.savefig("eda_outputs/energy_by_traffic.png")
plt.close()

# Rain impact on energy usage
plt.figure(figsize=(8, 6))
sns.boxplot(x='rain_flag', y='energy_kwh', data=df)
plt.title("Energy Usage: Rain vs No Rain")
plt.xticks([0, 1], ['No Rain', 'Rain'])
plt.savefig("eda_outputs/energy_rain_comparison.png")
plt.close()

# Season vs energy usage
plt.figure(figsize=(8, 6))
sns.boxplot(x='season', y='energy_kwh', data=df, order=["Winter", "Summer", "Monsoon", "Autumn"])
plt.title("Energy Usage by Season")
plt.savefig("eda_outputs/energy_by_season.png")
plt.close()

print("✅ EDA complete. All charts and summary saved in 'eda_outputs/' folder.")
