# PastCast ML Weather Prediction Model

## Quick Start

```bash
cd backend

# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (generates data + trains XGBoost + Random Forest)
python -m ml.train

# 3. Start the Flask server (now uses ML model)
python app.py
```

## Architecture

```
ml/
├── data_generator.py    # Synthetic weather data (15 global cities, ~10K samples)
├── predictor.py         # WeatherRainfallPredictor (XGBoost + RF baseline)
├── train.py             # End-to-end training pipeline
├── data/
│   └── weather_data.csv # Generated training dataset
└── models/
    └── rain_predictor.pkl  # Trained model (pickle)
```

## Model Details

| Property | Value |
|---|---|
| **Algorithm** | XGBoost Classifier |
| **Baseline** | Random Forest |
| **Features** | 16 (temperature, humidity, pressure, wind, clouds, dew point, etc.) |
| **Training Data** | 10,000 synthetic samples across 15 cities |
| **Split** | 70% train / 15% val / 15% test |

## Using Real Data

Replace `ml/data/weather_data.csv` with a real dataset containing these columns:

```
temperature, humidity, pressure, wind_speed, cloud_coverage, dew_point,
visibility, month, day_of_year, latitude, longitude, quarter, is_monsoon,
temp_lag1, humidity_lag1, rainfall_lag1, rain_occurred
```

Recommended sources:
- [Kaggle Weather Forecasting Data](https://www.kaggle.com/datasets/iamsouravbanerjee/weather-forecasting-data)
- [NOAA Climate Data](https://www.ncei.noaa.gov/)

Then re-train: `python -m ml.train`
