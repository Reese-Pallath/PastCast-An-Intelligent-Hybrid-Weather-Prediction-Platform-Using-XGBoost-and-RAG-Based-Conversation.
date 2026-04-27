
import numpy as np
import pandas as pd
import os

np.random.seed(42)

# (city, lat, lon, temp_mean, temp_std, hum_base, rain_months, max_delta, wind_mean, wind_std)
CLIMATE_PROFILES = [
    ("Mumbai",       19.08,  72.88, 28, 3,   72, [6,7,8,9],          5,  15, 5),
    ("Delhi",        28.61,  77.21, 25, 8,   55, [7,8,9],            7,  12, 5),
    ("Chennai",      13.08,  80.27, 30, 3,   68, [10,11,12],         4,  14, 5),
    ("Bangalore",    12.97,  77.59, 24, 3,   65, [5,6,7,8,9,10],     5,  11, 4),
    ("Kolkata",      22.57,  88.36, 27, 5,   70, [6,7,8,9],          5,  13, 5),
    ("Hyderabad",    17.39,  78.47, 28, 4,   60, [6,7,8,9,10],       6,  13, 4),
    ("Pune",         18.52,  73.86, 26, 4,   62, [6,7,8,9],          6,  12, 4),
    ("Jaipur",       26.91,  75.79, 26, 9,   48, [7,8],              8,  11, 4),
    ("Ahmedabad",    23.02,  72.57, 28, 6,   52, [7,8],              8,  14, 5),
    ("Kochi",         9.93,  76.26, 28, 2,   79, [6,7,8,9,10,11],   3,  13, 4),
    ("Bhopal",       23.26,  77.40, 26, 7,   55, [7,8,9],            7,  11, 4),
    ("Nagpur",       21.15,  79.09, 27, 6,   55, [6,7,8,9],          8,  12, 4),
    ("Lucknow",      26.85,  80.95, 25, 8,   57, [7,8,9],            7,  11, 4),
    ("Patna",        25.59,  85.14, 26, 7,   65, [7,8,9],            6,  11, 4),
    ("London",       51.51,  -0.13, 12, 6,   78, [10,11,12,1],       5,  14, 6),
    ("New York",     40.71, -74.01, 13, 10,  63, [4,5,6,7],          7,  13, 6),
    ("Tokyo",        35.68, 139.69, 16, 8,   65, [6,7,9],            6,  11, 5),
    ("Sydney",      -33.87, 151.21, 18, 5,   62, [1,2,3],            6,  16, 6),
    ("Dubai",        25.20,  55.27, 30, 7,   45, [12,1,2],           8,  17, 6),
    ("Singapore",     1.35, 103.82, 27, 1.5, 82, [11,12,1,2],        3,  12, 4),
    ("Bangkok",      13.75, 100.52, 29, 2,   74, [5,6,7,8,9,10],     4,  11, 4),
    ("Moscow",       55.76,  37.62,  6, 14,  72, [6,7,8],            6,  13, 6),
    ("Cairo",        30.04,  31.24, 23, 7,   45, [11,12,1],          8,  14, 6),
    ("Nairobi",      -1.29,  36.82, 20, 3,   60, [3,4,5,10,11],      5,  11, 4),
    ("Jakarta",      -6.21, 106.85, 27, 1,   80, [10,11,12,1,2,3],  3,  11, 4),
]

# Thresholds matching open_meteo_service.py and WeatherResults display
RAIN_MM_THRESHOLD   = 0.5
HEAT_C_THRESHOLD    = 35.0
WIND_KMH_THRESHOLD  = 40.0
CLOUD_PCT_THRESHOLD = 70.0
GOOD_TEMP_MAX       = 32.0
GOOD_WIND_MAX       = 30.0
GOOD_CLOUD_MAX      = 60.0


def generate_weather_dataset(n_samples: int = 50_000, output_path: str = None) -> pd.DataFrame:
    """
    Generate a synthetic weather dataset with realistic climate patterns.

    Produces a DataFrame with 16 feature columns plus 5 binary target columns:
    rain_occurred, extreme_heat, high_wind, cloudy, good_weather.
    """
    records = []
    samples_per_city = n_samples // len(CLIMATE_PROFILES)
    remainder = n_samples - samples_per_city * len(CLIMATE_PROFILES)

    for idx, (city, lat, lon, temp_mean, temp_std, hum_base,
              rain_months, max_delta, wind_mean, wind_std) in enumerate(CLIMATE_PROFILES):
        city_n = samples_per_city + (1 if idx < remainder else 0)

        for _ in range(city_n):
            day_of_year = np.random.randint(1, 366)
            month       = int(np.clip(np.ceil(day_of_year / 30.44), 1, 12))
            quarter     = (month - 1) // 3 + 1
            is_monsoon  = int(month in rain_months)

            # ── Temperature ─────────────────────────────────────
            seasonal_offset = temp_std * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            if lat < 0:
                seasonal_offset = -seasonal_offset
            temperature = temp_mean + seasonal_offset + np.random.normal(0, 2)

            # Daily max is 3–max_delta °C above mean
            temp_max = temperature + np.random.uniform(3, max_delta)

            # ── Humidity ────────────────────────────────────────
            humidity = hum_base + (18 if is_monsoon else -8) + np.random.normal(0, 8)
            humidity = float(np.clip(humidity, 15, 100))

            # ── Pressure ────────────────────────────────────────
            pressure = 1013.25 + np.random.normal(0, 5)
            if is_monsoon:
                pressure -= np.random.uniform(3, 10)

            # ── Wind ────────────────────────────────────────────
            wind_speed = abs(np.random.normal(wind_mean, wind_std))
            if is_monsoon:
                wind_speed += np.random.uniform(2, 10)
            # Occasional storm gusts (5% of monsoon days)
            if is_monsoon and np.random.rand() < 0.05:
                wind_speed += np.random.uniform(20, 40)
            wind_speed = float(np.clip(wind_speed, 0, 80))

            # ── Cloud cover ─────────────────────────────────────
            cloud_coverage = 25 + (45 if is_monsoon else 0) + np.random.normal(0, 18)
            cloud_coverage = float(np.clip(cloud_coverage, 0, 100))

            # ── Dew point & visibility ──────────────────────────
            dew_point  = temperature - ((100 - humidity) / 5) + np.random.normal(0, 1.5)
            visibility = 10 - (3 if is_monsoon else 0) + np.random.normal(0, 2)
            visibility = float(np.clip(visibility, 0.5, 20))

            # ── Lag features ────────────────────────────────────
            temp_lag1     = temperature + np.random.normal(0, 1.5)
            humidity_lag1 = float(np.clip(humidity + np.random.normal(0, 5), 15, 100))

            # ── Rain probability (physics-informed) ─────────────
            rp = 0.0
            rp += max(0.0, (humidity - 55)) / 90          # humidity signal
            rp += max(0.0, (1013 - pressure)) / 45        # low pressure
            rp += max(0.0, (cloud_coverage - 35)) / 110   # clouds
            rp += max(0.0, (dew_point - (temperature - 4))) / 8  # near dew point
            if is_monsoon:
                rp += 0.25
            if temp_max > 38:                              # convective rain
                rp += 0.08
            rp = float(np.clip(rp, 0.0, 0.95))

            rain_occurred = int(np.random.rand() < rp)
            rainfall_mm   = 0.0
            if rain_occurred:
                rainfall_mm = float(np.random.exponential(6) + RAIN_MM_THRESHOLD)

            rainfall_lag1 = float(max(0.0, rainfall_mm + np.random.normal(0, 2)))

            # ── Binary targets ───────────────────────────────────
            extreme_heat = int(temp_max     > HEAT_C_THRESHOLD)
            high_wind    = int(wind_speed   > WIND_KMH_THRESHOLD)
            cloudy       = int(cloud_coverage > CLOUD_PCT_THRESHOLD)
            good_weather = int(
                rainfall_mm  <= RAIN_MM_THRESHOLD
                and temp_max <= GOOD_TEMP_MAX
                and wind_speed <= GOOD_WIND_MAX
                and cloud_coverage <= GOOD_CLOUD_MAX
            )

            records.append({
                "city":          city,
                "latitude":      round(lat + np.random.normal(0, 0.3), 4),
                "longitude":     round(lon + np.random.normal(0, 0.3), 4),
                "day_of_year":   day_of_year,
                "month":         month,
                "quarter":       quarter,
                "is_monsoon":    is_monsoon,
                "temperature":   round(temperature, 2),
                "temp_max":      round(temp_max, 2),
                "humidity":      round(humidity, 2),
                "pressure":      round(pressure, 2),
                "wind_speed":    round(wind_speed, 2),
                "cloud_coverage": round(cloud_coverage, 2),
                "dew_point":     round(dew_point, 2),
                "visibility":    round(visibility, 2),
                "temp_lag1":     round(temp_lag1, 2),
                "humidity_lag1": round(humidity_lag1, 2),
                "rainfall_lag1": round(rainfall_lag1, 2),
                "rainfall_mm":   round(rainfall_mm, 2),
                # targets
                "rain_occurred": rain_occurred,
                "extreme_heat":  extreme_heat,
                "high_wind":     high_wind,
                "cloudy":        cloudy,
                "good_weather":  good_weather,
            })

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✅ Generated {len(df)} samples → {output_path}")
        for t in ("rain_occurred", "extreme_heat", "high_wind", "cloudy", "good_weather"):
            print(f"   {t}: {df[t].mean():.2%} positive rate")

    return df


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(script_dir, "data", "weather_data.csv")
    generate_weather_dataset(n_samples=50_000, output_path=out)
