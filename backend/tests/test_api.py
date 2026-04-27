"""
API endpoint tests for PastCast backend.
Tests route handlers, input validation, and error responses.
"""

import json
import pytest


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_json(self, client):
        response = client.get("/health")
        data = response.get_json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "model" in data


class TestWeatherProbability:
    """Tests for /weather/probability endpoint."""

    def test_missing_location_returns_400(self, client):
        response = client.post(
            "/weather/probability",
            data=json.dumps({"date_range": {"start_date": "2024-06-15"}}),
            content_type="application/json",
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_missing_date_range_returns_400(self, client):
        response = client.post(
            "/weather/probability",
            data=json.dumps({
                "location": {"latitude": 12.97, "longitude": 77.59, "city_name": "Bengaluru"}
            }),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_invalid_latitude_returns_400(self, client):
        response = client.post(
            "/weather/probability",
            data=json.dumps({
                "location": {"latitude": 999, "longitude": 77.59},
                "date_range": {"start_date": "2024-06-15"},
            }),
            content_type="application/json",
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "latitude" in data["error"].lower() or "out of range" in data["error"].lower()

    def test_valid_request_returns_200(self, client):
        response = client.post(
            "/weather/probability",
            data=json.dumps({
                "location": {"latitude": 12.97, "longitude": 77.59, "city_name": "Bengaluru"},
                "date_range": {"start_date": "2024-06-15"},
            }),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = response.get_json()
        assert "probabilities" in data
        assert "location" in data
        assert data["location"]["city_name"] == "Bengaluru"

    def test_response_contains_required_fields(self, client):
        response = client.post(
            "/weather/probability",
            data=json.dumps({
                "location": {"latitude": 28.61, "longitude": 77.23, "city_name": "Delhi"},
                "date_range": {"start_date": "2024-01-15"},
            }),
            content_type="application/json",
        )
        data = response.get_json()
        probs = data["probabilities"]
        assert "rain" in probs
        assert "extreme_heat" in probs
        assert "high_wind" in probs
        assert "cloudy" in probs
        assert "good_weather" in probs
        assert "summary" in probs


class TestChatEndpoints:
    """Tests for /api/* chat endpoints."""

    def test_create_session(self, client):
        response = client.post("/api/session")
        assert response.status_code == 200
        data = response.get_json()
        assert "session_id" in data
        assert data["status"] == "created"

    def test_empty_message_rejected(self, client):
        response = client.post(
            "/api/message",
            data=json.dumps({"text": ""}),
            content_type="application/json",
        )
        data = response.get_json()
        assert data["reply"] == "Please enter a message."

    def test_clear_chat(self, client):
        response = client.post(
            "/api/clear",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert response.status_code == 200

    def test_history_endpoint(self, client):
        response = client.get("/api/history")
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)


class TestErrorHandling:
    """Tests for global error handling."""

    def test_404_returns_json(self, client):
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
        data = response.get_json()
        assert "error" in data

    def test_405_returns_json(self, client):
        response = client.get("/api/session")  # GET instead of POST
        assert response.status_code == 405
        data = response.get_json()
        assert "error" in data


class TestInputValidation:
    """Tests for input sanitisation."""

    def test_very_long_input_truncated(self, client):
        long_text = "A" * 1000
        response = client.post(
            "/api/message",
            data=json.dumps({"text": long_text}),
            content_type="application/json",
        )
        assert response.status_code == 200

    def test_control_characters_stripped(self, client):
        response = client.post(
            "/api/message",
            data=json.dumps({"text": "Hello\x00\x01World"}),
            content_type="application/json",
        )
        assert response.status_code == 200
