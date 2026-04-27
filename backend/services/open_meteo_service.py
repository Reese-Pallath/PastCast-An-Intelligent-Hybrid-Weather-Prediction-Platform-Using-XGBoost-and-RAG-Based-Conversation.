"""
Open-Meteo Historical Weather Service

Fetches actual measured historical data from the Open-Meteo archive API
(free, no API key required, ERA5 reanalysis data since 1940).
Results are cached in-memory for 24 hours so identical requests always
return identical numbers.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

_cache: dict = {}
_CACHE_TTL = 86400  # 24 hours

_DAILY_VARS = (
    "precipitation_sum,"
    "temperature_2m_max,"
    "temperature_2m_mean,"
    "windspeed_10m_max,"
    "cloudcover_mean"
)

# Thresholds used for all probability calculations — kept constant so
# numbers are comparable across locations and dates.
_RAIN_MM_THRESHOLD   = 0.5   # mm/day → rainy day
_HEAT_C_THRESHOLD    = 35.0  # °C max → extreme heat day
_WIND_KMH_THRESHOLD  = 40.0  # km/h max → high wind day
_CLOUD_PCT_THRESHOLD = 70.0  # % → cloudy day
_GOOD_TEMP_MAX       = 32.0  # °C — comfortable upper bound
_GOOD_WIND_MAX       = 30.0  # km/h — comfortable upper bound
_GOOD_CLOUD_MAX      = 60.0  # % — partly cloudy or better


def _build_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s


_session = _build_session()


def _fetch_archive(lat: float, lng: float, start: str, end: str) -> dict | None:
    try:
        resp = _session.get(
            ARCHIVE_URL,
            params={
                "latitude": lat,
                "longitude": lng,
                "start_date": start,
                "end_date": end,
                "daily": _DAILY_VARS,
                "timezone": "auto",
                "wind_speed_unit": "kmh",
            },
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Open-Meteo request failed (%s – %s): %s", start, end, exc)
        return None


def _compute_stats(daily: dict) -> dict | None:
    precip  = [v if v is not None else 0.0 for v in daily.get("precipitation_sum", [])]
    t_max   = [v if v is not None else 25.0 for v in daily.get("temperature_2m_max", [])]
    t_mean  = [v if v is not None else 20.0 for v in daily.get("temperature_2m_mean", [])]
    wind    = [v if v is not None else 0.0 for v in daily.get("windspeed_10m_max", [])]
    cloud   = [v if v is not None else 0.0 for v in daily.get("cloudcover_mean", [])]

    n = len(precip)
    if n == 0:
        return None

    rain_days  = sum(1 for p in precip if p > _RAIN_MM_THRESHOLD)
    heat_days  = sum(1 for t in t_max  if t > _HEAT_C_THRESHOLD)
    wind_days  = sum(1 for w in wind   if w > _WIND_KMH_THRESHOLD)
    cloud_days = sum(1 for c in cloud  if c > _CLOUD_PCT_THRESHOLD)
    good_days  = sum(
        1 for i in range(n)
        if (precip[i] <= _RAIN_MM_THRESHOLD
            and t_max[i] <= _GOOD_TEMP_MAX
            and wind[i] <= _GOOD_WIND_MAX
            and cloud[i] <= _GOOD_CLOUD_MAX)
    )

    return {
        "rain_prob":   round(rain_days  / n * 100, 1),
        "heat_prob":   round(heat_days  / n * 100, 1),
        "wind_prob":   round(wind_days  / n * 100, 1),
        "cloud_prob":  round(cloud_days / n * 100, 1),
        "good_prob":   round(good_days  / n * 100, 1),
        "avg_temp":    round(sum(t_mean) / n, 1),
        "avg_wind":    round(sum(wind)   / n, 1),
        "avg_cloud":   round(sum(cloud)  / n, 1),
        "data_points": n,
    }


def _merge_stats(stats_list: list[dict]) -> dict:
    numeric_keys = [
        "rain_prob", "heat_prob", "wind_prob", "cloud_prob",
        "good_prob", "avg_temp", "avg_wind", "avg_cloud",
    ]
    merged: dict = {}
    for k in numeric_keys:
        vals = [s[k] for s in stats_list if k in s]
        merged[k] = round(sum(vals) / len(vals), 1) if vals else 0.0
    merged["data_points"] = sum(s.get("data_points", 0) for s in stats_list)
    return merged


def get_historical_probabilities(
    lat: float,
    lng: float,
    start_date: str,
    end_date: str,
) -> dict | None:
    """
    Return deterministic weather probabilities derived from actual
    historical measurements.

    Strategy:
    - Dates fully in the past (> 5 days ago): fetch the exact range.
    - Recent / future dates: average the same calendar window across
      the last 5 available years (climatological mean).

    Returns None only if the network is completely unavailable.
    Cached for 24 hours — same inputs always produce identical output.
    """
    lat_r = round(lat, 2)
    lng_r = round(lng, 2)
    cache_key = (lat_r, lng_r, start_date, end_date)

    now_ts = time.time()
    if cache_key in _cache:
        result, ts = _cache[cache_key]
        if now_ts - ts < _CACHE_TTL:
            return result

    start_dt = datetime.fromisoformat(start_date).date()
    end_dt   = datetime.fromisoformat(end_date).date()
    # Open-Meteo archive lags by ~5 days
    archive_cutoff = date.today() - timedelta(days=6)

    if end_dt <= archive_cutoff:
        # Direct historical fetch
        raw = _fetch_archive(lat_r, lng_r, start_date, end_date)
        if raw is None:
            return None
        result = _compute_stats(raw.get("daily", {}))
        climatological = False
    else:
        # Climatological fallback — fetch the same calendar window across the
        # last 5 available years IN PARALLEL so we stay well within the
        # frontend's 30-second request timeout.
        span = (end_dt - start_dt).days
        fetch_ranges: list[tuple[str, str]] = []
        for years_back in range(1, 6):
            try:
                h_start = start_dt.replace(year=start_dt.year - years_back)
            except ValueError:
                h_start = start_dt.replace(year=start_dt.year - years_back, day=28)
            h_end = h_start + timedelta(days=span)
            if h_end <= archive_cutoff:
                fetch_ranges.append((str(h_start), str(h_end)))

        stats_list: list[dict] = []
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {
                pool.submit(_fetch_archive, lat_r, lng_r, s, e): (s, e)
                for s, e in fetch_ranges
            }
            for future in as_completed(futures, timeout=25):
                try:
                    raw = future.result()
                    if raw is not None:
                        s = _compute_stats(raw.get("daily", {}))
                        if s:
                            stats_list.append(s)
                except Exception as exc:
                    logger.warning("Climatological fetch failed: %s", exc)

        if not stats_list:
            return None
        result = _merge_stats(stats_list)
        climatological = True

    if result is None:
        return None

    result["climatological"] = climatological
    _cache[cache_key] = (result, now_ts)
    return result
