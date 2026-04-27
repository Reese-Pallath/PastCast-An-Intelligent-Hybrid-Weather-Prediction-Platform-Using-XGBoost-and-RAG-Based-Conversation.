"""
Unit tests for the ML weather prediction model.

Run:
    cd backend
    python -m pytest ml/test_ml_model.py -v
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.data_generator import generate_weather_dataset
from ml.predictor import WeatherRainfallPredictor, FEATURE_COLS

def test_data_generator_shape():
    df = generate_weather_dataset(n_samples=500)
    assert len(df) == 500, f"Expected 500 rows, got {len(df)}"

def test_data_generator_columns():
    df = generate_weather_dataset(n_samples=100)
    for col in FEATURE_COLS:
        assert col in df.columns, f"Missing column: {col}"
    assert "rain_occurred" in df.columns

def test_data_generator_target_binary():
    df = generate_weather_dataset(n_samples=200)
    unique_vals = set(df["rain_occurred"].unique())
    assert unique_vals.issubset({0, 1}), f"Target should be 0/1, got {unique_vals}"

def test_data_generator_rain_ratio():
    """Rain should occur between 10% and 90% of the time (sanity check)."""
    df = generate_weather_dataset(n_samples=2000)
    ratio = df["rain_occurred"].mean()
    assert 0.10 < ratio < 0.90, f"Unrealistic rain ratio: {ratio:.2%}"

class TestPredictor:

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Train a small model for testing."""
        self.df = generate_weather_dataset(n_samples=1000)
        self.pred = WeatherRainfallPredictor()
        self.df = self.pred.preprocess(self.df)
        splits = self.pred.split_data(self.df)
        self.X_train, self.X_val, self.X_test = splits[0], splits[1], splits[2]
        self.y_train, self.y_val, self.y_test = splits[3], splits[4], splits[5]
        self.pred.train_xgboost(self.X_train, self.X_val, self.y_train, self.y_val)
        self.model_path = str(tmp_path / "test_model.pkl")

    def test_training_completes(self):
        assert self.pred.model is not None

    def test_prediction_keys(self):
        sample = {f: 0.0 for f in FEATURE_COLS}
        sample.update({"temperature": 28, "humidity": 75, "pressure": 1010,
                        "month": 7, "is_monsoon": 1, "latitude": 19.0})
        result = self.pred.predict_rain(sample)
        assert "will_rain" in result
        assert "rain_probability" in result
        assert "confidence" in result

    def test_probability_range(self):
        sample = {f: 0.0 for f in FEATURE_COLS}
        sample.update({"temperature": 30, "humidity": 80, "pressure": 1005,
                        "month": 7, "is_monsoon": 1, "latitude": 19.0})
        result = self.pred.predict_rain(sample)
        assert 0 <= result["rain_probability"] <= 1
        assert 0.5 <= result["confidence"] <= 1.0

    def test_save_and_load(self):
        self.pred.save(self.model_path)
        loaded = WeatherRainfallPredictor.load(self.model_path)
        assert loaded.model is not None
        sample = {f: 0.0 for f in FEATURE_COLS}
        result = loaded.predict_rain(sample)
        assert "rain_probability" in result

    def test_evaluate_metrics(self):
        metrics = self.pred.evaluate(
            self.pred.model, self.X_test, self.y_test, label="Test"
        )
        assert metrics["Accuracy"] > 0.5, "Model should beat random chance"
        assert all(k in metrics for k in ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"])

    def test_feature_importance(self):
        fi = self.pred.feature_importance(top_n=5)
        assert fi is not None
        assert len(fi) == 5

    def test_baseline_random_forest(self):
        self.pred.train_random_forest(self.X_train, self.y_train)
        assert self.pred.baseline_model is not None
        metrics = self.pred.evaluate(
            self.pred.baseline_model, self.X_test, self.y_test, label="RF"
        )
        assert metrics["Accuracy"] > 0.5
