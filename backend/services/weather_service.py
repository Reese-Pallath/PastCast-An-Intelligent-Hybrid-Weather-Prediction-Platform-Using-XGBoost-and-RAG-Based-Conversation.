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

def compute_wsi(temp: float, rain_prob: float, wind_speed: float, cloud_coverage: float) -> tuple:
    """
    Compute Weather Suitability Index.

    Returns (wsi_percent, wsi_label).
    """
    heat_risk = max(0.0, min(1.0, (temp - 25) / 15.0))
    wind_risk = max(0.0, min(1.0, (wind_speed - 20) / 200.0))
    rain_factor = rain_prob / 100.0
    clear_sky = (100.0 - cloud_coverage) / 100.0

    score = (
        (1 - heat_risk) * 0.3
        + (1 - rain_factor) * 0.3
        + (1 - wind_risk) * 0.2
        + clear_sky * 0.2
    )
    wsi = max(0.0, min(100.0, score * 100))

    # Heat penalty
    if temp >= 38:
        wsi *= 0.4
    elif temp >= 33:
        wsi *= 0.7

    if wsi >= 80:
        label = "Excellent"
    elif wsi >= 60:
        label = "Good"
    elif wsi >= 40:
        label = "Fair"
    else:
        label = "Poor"

    return round(wsi, 2), label


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
