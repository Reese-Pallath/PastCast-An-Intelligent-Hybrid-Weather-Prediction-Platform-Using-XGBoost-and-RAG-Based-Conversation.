from __future__ import annotations

"""
Weather Service — Business logic for weather probability calculations.

Extracts all weather-related computation from the monolithic app.py so it
can be tested independently and reused across endpoints.
"""

import logging
from datetime import datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ── HTTP session with retry ─────────────────────────────────────
_http_session = None


def _get_http_session() -> requests.Session:
    """Return a requests.Session with automatic retry & backoff."""
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retries)
        _http_session.mount("https://", adapter)
        _http_session.mount("http://", adapter)
    return _http_session


# ── Live weather fetching ───────────────────────────────────────

DEFAULTS = {
    "temperature": 25, "humidity": 60, "pressure": 1013,
    "wind_speed": 8, "cloud_coverage": 50, "dew_point": 15,
    "visibility": 10, "rain_1h": 0,
}


def fetch_live_weather(
    city_name: str,
    lat: float,
    lng: float,
    api_key: str | None,
    weather_url: str,
) -> dict:
    """Fetch current weather from OpenWeatherMap."""
    if not api_key:
        logger.warning("OpenWeather API key missing — using defaults")
        return dict(DEFAULTS)
    try:
        session = _get_http_session()
        r = session.get(
            f"{weather_url}/weather",
            params={"lat": lat, "lon": lng, "appid": api_key, "units": "metric"},
            timeout=8,
        ).json()
        if "main" not in r:
            logger.warning("No 'main' key in OpenWeather response for %s", city_name)
            return dict(DEFAULTS)
        temp = r["main"]["temp"]
        humidity = r["main"]["humidity"]
        return {
            "temperature": temp,
            "humidity": humidity,
            "pressure": r["main"].get("pressure", 1013),
            "wind_speed": r.get("wind", {}).get("speed", 8),
            "cloud_coverage": r.get("clouds", {}).get("all", 50),
            "dew_point": temp - ((100 - humidity) / 5),
            "visibility": r.get("visibility", 10000) / 1000,
            "rain_1h": r.get("rain", {}).get("1h", 0),
        }
    except requests.RequestException as exc:
        logger.error("OpenWeather API error for %s: %s", city_name, exc)
        return dict(DEFAULTS)


# ── Label helpers ───────────────────────────────────────────────

def heat_label(temp: float) -> str:
    if temp < 25:
        return "Cool"
    elif temp <= 30:
        return "Pleasant"
    elif temp <= 35:
        return "Warm"
    elif temp <= 40:
        return "Hot"
    return "Extreme Heat"


def wind_label(speed: float) -> str:
    if speed <= 10:
        return "Calm"
    elif speed <= 20:
        return "Light"
    elif speed <= 30:
        return "Moderate"
    elif speed <= 40:
        return "Strong"
    return "Very Strong"


def cloud_label(coverage: float) -> str:
    if coverage <= 20:
        return "Clear"
    elif coverage <= 50:
        return "Partly Cloudy"
    elif coverage <= 80:
        return "Cloudy"
    return "Overcast"


def rain_label(probability: float) -> str:
    if probability <= 20:
        return "Very Low"
    elif probability <= 40:
        return "Low"
    elif probability <= 70:
        return "Moderate"
    return "High"


# ── Weather Suitability Index ───────────────────────────────────

def classify_weather_score(score: float) -> str:
    """Map a 0–100 weather pleasantness score to a human-readable label."""
    if score >= 75:
        return "Excellent"
    elif score >= 55:
        return "Good"
    elif score >= 35:
        return "Moderate"
    else:
        return "Poor"


def compute_wsi(temp: float, rain_prob: float, wind_speed: float, cloud_coverage: float) -> tuple:
    """
    Compute Weather Suitability Index (0–100) and a quality label.

    Uses the same weighted additive formula as _good_score so that the
    numeric score and its label are always consistent.

    Weights:  rain 35% | cloud 25% | temperature comfort 25% | wind calmness 15%

    Returns (wsi_percent, wsi_label).
    """
    # Rain and cloud are NEGATIVE factors — invert so high = bad
    rain_score  = 1.0 - (rain_prob / 100.0)
    cloud_score = 1.0 - (cloud_coverage / 100.0)

    # Temperature comfort: 18–28 °C ideal (score = 1.0)
    if 18.0 <= temp <= 28.0:
        temp_score = 1.0
    elif temp < 18.0:
        temp_score = max(0.0, 1.0 - (18.0 - temp) / 18.0)
    else:
        temp_score = max(0.0, 1.0 - (temp - 28.0) / 20.0)

    # Wind calmness: ≤15 km/h ideal, reaches 0 at 75 km/h+
    wind_score = max(0.0, 1.0 - max(0.0, wind_speed - 15.0) / 60.0)

    wsi = (
        rain_score  * 0.35
        + cloud_score * 0.25
        + temp_score  * 0.25
        + wind_score  * 0.15
    ) * 100.0

    wsi = round(max(0.0, min(100.0, wsi)), 2)
    return wsi, classify_weather_score(wsi)


# ── Seasonal fallback ──────────────────────────────────────────

def get_seasonal_base(lat: float, date_str: str) -> dict:
    """Rule-based seasonal weather when ML model is unavailable."""
    mo = datetime.fromisoformat(date_str).month - 1
    is_north = lat > 0
    if is_north:
        if 2 <= mo <= 4:
            return {"rain": 25, "sunny": 60, "cloudy": 40, "temp": 28}
        if 5 <= mo <= 7:
            return {"rain": 70, "sunny": 45, "cloudy": 60, "temp": 32}
        if 8 <= mo <= 10:
            return {"rain": 50, "sunny": 55, "cloudy": 45, "temp": 26}
        return {"rain": 15, "sunny": 70, "cloudy": 30, "temp": 20}
    else:
        if 8 <= mo <= 10:
            return {"rain": 25, "sunny": 60, "cloudy": 40, "temp": 28}
        if mo >= 11 or mo <= 1:
            return {"rain": 70, "sunny": 45, "cloudy": 60, "temp": 32}
        if 2 <= mo <= 4:
            return {"rain": 50, "sunny": 55, "cloudy": 45, "temp": 26}
        return {"rain": 15, "sunny": 70, "cloudy": 30, "temp": 20}


def generate_condition(base_prob: float, label: str, threshold: str, description: str) -> dict:
    """Generate a deterministic weather condition for rule-based fallback."""
    val = max(0.0, min(100.0, base_prob))
    return {
        "probability": round(val, 2),
        "label": label,
        "threshold": threshold,
        "description": description,
    }


def make_condition(prob: float, label: str, threshold: str, desc: str) -> dict:
    """Create a weather condition dict (for ML-based results)."""
    return {
        "probability": round(prob, 2),
        "label": label,
        "threshold": threshold,
        "description": desc,
    }
