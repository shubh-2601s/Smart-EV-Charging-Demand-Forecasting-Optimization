
# ⚡ Smart-EV-Charging-Demand-Forecasting-Optimization

---

## 📘 Project Overview

With the growing adoption of electric vehicles (EVs), smart infrastructure planning is critical. This project aims to forecast EV charging demand using weather, time, and traffic data. By combining advanced data engineering with AI models like Facebook Prophet, the system empowers stakeholders to make informed decisions on station management, energy load distribution, and mobility optimization.

This solution integrates multiple real-world datasets and builds a forecasting pipeline with visual outputs—making.

---

## 🎯 Key Objectives

- 📊 **Data Integration:** Merge EV usage, weather, and traffic datasets by timestamp and station.
- 🧠 **Feature Engineering:** Extract temporal, environmental, and traffic-based features affecting charging behavior.
- 🔍 **EDA:** Visualize trends and detect correlations in energy usage across stations.
- 📈 **Forecasting:** Build both station-wise and aggregate forecasting models using Prophet.
- 💡 **Optimization Insight:** Enable strategy development for smarter EV station management.

---

## 🧰 Tech Stack

| Tool / Library     | Purpose                                            |
|--------------------|----------------------------------------------------|
| Python             | Data processing, feature engineering, modeling     |
| Pandas, NumPy      | Data manipulation                                  |
| Matplotlib, Seaborn| Visualization and analysis                         |
| Facebook Prophet   | Time-series forecasting                            |
| CSV Files          | Lightweight data storage                           |
| *(Optional)* Tableau / Streamlit | Dashboard visualization             |
| *(Optional)* APIs  | Real-time traffic and weather input (OpenWeather, Google Maps, etc.) |

---

## 📁 Repository Structure

| File / Folder                | Description                                                                  |
|------------------------------|------------------------------------------------------------------------------|
| `merge_ds.py`                | Merges EV usage, weather, and traffic datasets by time/location.             |
| `feature_extractions.py`     | Engineers relevant features (time, weather, traffic, lag).                   |
| `Eda_Ev.py`                  | Performs exploratory data analysis (EDA) and saves visuals.                  |
| `forecasting_ev_demand.py`   | Trains station-wise Prophet models and saves predictions.                    |
| `forecasting_ev.py`          | Builds a global forecasting model using multiple regressors.                |
| `eda_outputs/`               | Stores summary plots, heatmaps, and usage trends.                           |
| `forecasts/`                 | Contains forecast data and plots per station.                               |
| `ev_usage.csv`               | Sample EV charging station data.                                            |
| `weather.csv`                | Sample weather conditions data.                                             |
| `traffic.csv`                | Sample traffic flow data.                                                   |

---

## 📂 Dataset Overview

**1. EV Usage Data (`ev_usage.csv`)**
- Contains timestamped energy consumption data (kWh) for each charging station.

**2. Weather Data (`weather.csv`)**
- Hourly temperature, rainfall, humidity, etc., joined on timestamp and station location.

**3. Traffic Data (`traffic.csv`)**
- Hourly vehicle counts and congestion indicators for major road segments near stations.

---

## ⚙️ How to Run

### 1. Merge Datasets
```bash
python merge_ds.py
````

🔹 Output: `full_merged_data.csv`

---

### 2. Feature Engineering

```bash
python feature_extractions.py
```

🔹 Output: `full_merged_data_featured.csv`

---

### 3. Exploratory Data Analysis

```bash
python Eda_Ev.py
```

🔹 Output Directory: `eda_outputs/`
(Heatmaps, boxplots, hourly/seasonal usage patterns, etc.)

---

### 4. Station-wise Forecasting

```bash
python forecasting_ev_demand.py
```

🔹 Output Directory: `forecasts/`
(Forecast CSV + PNG for each station, next 7-day prediction)

---

### 5. Global Forecasting (With Regressors)

```bash
python forecasting_ev.py
```

🔹 Output: `forecast_results.csv` + printed MAE, RMSE

---

## 📤 Output Summary

| File / Folder                   | Description                                             |
| ------------------------------- | ------------------------------------------------------- |
| `full_merged_data.csv`          | Combined dataset from EV, weather, and traffic sources. |
| `full_merged_data_featured.csv` | Dataset with engineered features.                       |
| `eda_outputs/`                  | Visual insights (EDA)                                   |
| `forecasts/`                    | CSV + plots of station-wise predictions                 |
| `forecast_results.csv`          | Forecast output from the global model                   |

---

## DashBoard Screenshot
![EV Charging Demand Patterns Temporal, Traffic   Seasonal Insights dashboard](https://github.com/user-attachments/assets/a5f5028a-7200-41c2-9466-7d373c9f0bae)


## 📈 Technical Highlights

* **Lag Features**: Incorporate past usage for improved accuracy.
* **Categorical Mapping**: Label encoding for weekdays, seasons, and weather.
* **Temporal Trends**: Capture hourly/weekly patterns via cyclical encoding.
* **Multivariate Forecasting**: Include external regressors in Prophet models.

---

## 🚀 Future Enhancements

This project can be further expanded with:

### 📱 Smart User Experience

* **Mobile App Interface** for users to find least busy charging stations in real-time.
* **Push Notifications** for charging slot availability, best time-to-charge alerts.


### ⚡ Charging Optimization

* Recommend **optimal charging station deployment** based on forecasted high-demand zones.
* Suggest **load balancing strategies** to reduce grid pressure during peak usage.

### 🌍 Sustainability & Policy Insights

* Visualize **carbon offset impact** of optimized charging.
* Provide **urban planners with simulations** for new EV infrastructure locations.


---



## 🧑‍💻 Contact

For support, ideas, or collaboration:

* GitHub: [https://github.com/shubh-2601s](https://github.com/shubh-2601s)
* Email: [10221shubham.s@gmail.com](mailto:10221shubham.s@gmail.com)

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---
