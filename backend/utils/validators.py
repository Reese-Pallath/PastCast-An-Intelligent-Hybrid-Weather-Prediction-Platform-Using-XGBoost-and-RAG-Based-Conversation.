from __future__ import annotations

"""
Input validation and sanitisation utilities.
Applied at the API boundary to reject malformed or dangerous input.
"""

import re
import logging

logger = logging.getLogger(__name__)

# Maximum allowed length for user chat input
MAX_INPUT_LENGTH = 500

# Maximum allowed length for city name
MAX_CITY_LENGTH = 100


def sanitise_text(text: str, max_length: int = MAX_INPUT_LENGTH) -> str:
    """
    Sanitise user-provided text:
    - Strip leading / trailing whitespace
    - Remove control characters and null bytes
    - Enforce length limit
    """
    if not isinstance(text, str):
        return ""

    # Remove null bytes and control characters (keep newlines, tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = text.strip()

    if len(text) > max_length:
        logger.warning(
            "Input truncated: %d chars → %d chars", len(text), max_length
        )
        text = text[:max_length]

    return text


def validate_location(location: dict) -> tuple:
    """
    Validate location payload from /weather/probability.

    Returns
    -------
    (is_valid: bool, error_message: str | None)
    """
    if not isinstance(location, dict):
        return False, "location must be an object"

    lat = location.get("latitude")
    lng = location.get("longitude")

    try:
        lat = float(lat) if lat is not None else None
        lng = float(lng) if lng is not None else None
    except (TypeError, ValueError):
        return False, "latitude and longitude must be numbers"

    if lat is None or lng is None:
        return False, "latitude and longitude are required"

    if not (-90 <= lat <= 90):
        return False, f"latitude {lat} out of range [-90, 90]"

    if not (-180 <= lng <= 180):
        return False, f"longitude {lng} out of range [-180, 180]"

    city_name = location.get("city_name", "")
    if city_name and len(str(city_name)) > MAX_CITY_LENGTH:
        location["city_name"] = str(city_name)[:MAX_CITY_LENGTH]

    return True, None


def validate_date_range(date_range: dict) -> tuple:
    """
    Validate date_range payload.

    Returns
    -------
    (is_valid: bool, error_message: str | None)
    """
    if not isinstance(date_range, dict):
        return False, "date_range must be an object"

    start_date = date_range.get("start_date")
    if not start_date:
        return False, "date_range.start_date is required"

    # Basic ISO date format check
    iso_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}")
    if not iso_pattern.match(str(start_date)):
        return False, "start_date must be in ISO format (YYYY-MM-DD)"

    end_date = date_range.get("end_date")
    if end_date and not iso_pattern.match(str(end_date)):
        return False, "end_date must be in ISO format (YYYY-MM-DD)"

    return True, None
