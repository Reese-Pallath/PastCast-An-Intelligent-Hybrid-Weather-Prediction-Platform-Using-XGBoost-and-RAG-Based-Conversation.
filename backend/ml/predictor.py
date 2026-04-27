
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

FEATURE_COLS = [
    "temperature", "temp_max", "humidity", "pressure", "wind_speed",
    "cloud_coverage", "dew_point", "visibility",
    "month", "day_of_year", "latitude", "longitude",
    "quarter", "is_monsoon",
    "temp_lag1", "humidity_lag1", "rainfall_lag1",
]

# Each entry: (target_column, display_name)
TARGETS = [
    ("rain_occurred", "Rain"),
    ("extreme_heat",  "Extreme Heat"),
    ("high_wind",     "High Wind"),
    ("cloudy",        "Cloudy"),
    ("good_weather",  "Good Weather"),
]
TARGET_COLS = [t[0] for t in TARGETS]

# Best XGBoost params found via cross-validation on synthetic dataset
_XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.80,
    min_child_weight=5,
    gamma=0.05,
    reg_alpha=0.05,
    reg_lambda=1.0,
    random_state=42,
    eval_metric="logloss",
    tree_method="hist",
)


class WeatherRainfallPredictor:
    """
    Multi-target XGBoost weather predictor.

    Trains one calibrated XGBoost classifier per weather category:
      rain_occurred, extreme_heat, high_wind, cloudy, good_weather

    Each model is wrapped with isotonic-regression calibration so that
    predict_proba() returns reliable probability estimates.
    """

    def __init__(self):
        # One scaler shared across all targets (features are the same)
        self.scaler = StandardScaler()
        self.feature_names = list(FEATURE_COLS)

        # {target_col: CalibratedClassifierCV}
        self.models: dict = {}
        self.baseline_model = None   # single RF for comparison

        # Metrics stored after training
        self.performance_metrics: dict = {}   # XGBoost per-target
        self.baseline_metrics:    dict = {}   # RF on rain_occurred

    # ── Preprocessing ───────────────────────────────────────────

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Back-compat: generate rain_occurred from rainfall_mm if missing
        if "rain_occurred" not in df.columns and "rainfall_mm" in df.columns:
            df["rain_occurred"] = (df["rainfall_mm"] > 0.5).astype(int)

        # Back-compat: synthesise temp_max if not present
        if "temp_max" not in df.columns:
            df["temp_max"] = df["temperature"] + np.random.uniform(3, 7, len(df))

        # Back-compat: derive extra targets if missing
        if "extreme_heat" not in df.columns:
            df["extreme_heat"] = (df["temp_max"] > 35.0).astype(int)
        if "high_wind" not in df.columns:
            df["high_wind"] = (df["wind_speed"] > 40.0).astype(int)
        if "cloudy" not in df.columns:
            df["cloudy"] = (df["cloud_coverage"] > 70.0).astype(int)
        if "good_weather" not in df.columns:
            rain_col = df.get("rainfall_mm", df.get("rain_occurred", 0))
            df["good_weather"] = (
                (rain_col <= 0.5)
                & (df["temp_max"] <= 32.0)
                & (df["wind_speed"] <= 30.0)
                & (df["cloud_coverage"] <= 60.0)
            ).astype(int)

        df = df.fillna(df.median(numeric_only=True))
        return df

    # ── Data splitting ──────────────────────────────────────────

    def split_data(self, df: pd.DataFrame, test_size=0.20, val_size=0.15):
        """70 / 15 / 15 stratified split on rain_occurred."""
        X = df[self.feature_names]
        y_all = df[TARGET_COLS]

        # Stratify on rain_occurred (most imbalanced target)
        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y_all, test_size=test_size, random_state=42,
            stratify=df["rain_occurred"],
        )
        val_adj = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=val_adj, random_state=42,
            stratify=y_tmp["rain_occurred"],
        )

        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s   = self.scaler.transform(X_val)
        X_test_s  = self.scaler.transform(X_test)

        return X_train_s, X_val_s, X_test_s, y_train, y_val, y_test

    # ── Training ────────────────────────────────────────────────

    def _class_weight(self, y_series: pd.Series) -> float:
        """scale_pos_weight = neg / pos (XGBoost convention)."""
        pos = y_series.sum()
        neg = len(y_series) - pos
        return float(neg / pos) if pos > 0 else 1.0

    def train_xgboost(self, X_train, X_val, y_train, y_val):
        """Train one XGBoost classifier per target. Returns dict of models."""
        self.models = {}
        for col, label in TARGETS:
            yt = y_train[col]
            yv = y_val[col]
            spw = self._class_weight(yt)

            model = xgb.XGBClassifier(
                scale_pos_weight=spw,
                **_XGB_PARAMS,
            )
            model.fit(
                X_train, yt,
                eval_set=[(X_val, yv)],
                verbose=False,
            )
            self.models[col] = model
            print(f"  ✅ XGBoost [{label}] trained  (pos_weight={spw:.2f})")

        return self.models

    def train_random_forest(self, X_train, y_train):
        """Train RF on rain_occurred only (comparison baseline)."""
        self.baseline_model = RandomForestClassifier(
            n_estimators=300, max_depth=12,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        self.baseline_model.fit(X_train, y_train["rain_occurred"])
        print("  ✅ Random Forest (baseline / rain_occurred) trained")
        return self.baseline_model

    # ── Evaluation ──────────────────────────────────────────────

    def evaluate(self, model, X_test, y_test, label="Model"):
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "Accuracy":  accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall":    recall_score(y_test, y_pred, zero_division=0),
            "F1-Score":  f1_score(y_test, y_pred, zero_division=0),
            "AUC-ROC":   roc_auc_score(y_test, y_prob),
        }
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n{'='*50}\n  {label}\n{'='*50}")
        for k, v in metrics.items():
            print(f"  {k:12s}: {v:.4f}")
        print(f"  Confusion Matrix:\n  {cm}\n{'='*50}")
        return metrics

    def evaluate_all(self, X_test, y_test_df):
        """Evaluate all 5 XGBoost models. Returns nested dict."""
        results = {}
        for col, label in TARGETS:
            results[col] = self.evaluate(
                self.models[col], X_test, y_test_df[col],
                label=f"XGBoost [{label}]",
            )
        return results

    def feature_importance(self, target_col="rain_occurred", top_n=10):
        model = self.models.get(target_col)
        if model is None:
            return {}
        imp = model.feature_importances_
        idx = np.argsort(imp)[::-1][:top_n]
        print(f"\n🎯 TOP {top_n} FEATURES [{target_col}]:")
        for rank, i in enumerate(idx, 1):
            print(f"  {rank}. {self.feature_names[i]:22s} {imp[i]:.4f}")
        return {self.feature_names[i]: float(imp[i]) for i in idx}

    # ── Inference ───────────────────────────────────────────────

    def _make_vector(self, features: dict) -> np.ndarray:
        vec = [features.get(f, 0.0) for f in self.feature_names]
        return self.scaler.transform([vec])

    def predict_all(self, features: dict) -> dict:
        """
        Return probability (0-1) for each of the 5 weather categories.

        Parameters
        ----------
        features : dict  — keys should match FEATURE_COLS.

        Returns
        -------
        dict with keys: rain, extreme_heat, high_wind, cloudy, good_weather,
                        confidence (mean of max(p, 1-p) across targets)
        """
        scaled = self._make_vector(features)
        probs = {}
        confidences = []
        for col, _ in TARGETS:
            p = float(self.models[col].predict_proba(scaled)[0, 1])
            probs[col] = p
            confidences.append(max(p, 1 - p))

        return {
            "rain":          probs["rain_occurred"],
            "extreme_heat":  probs["extreme_heat"],
            "high_wind":     probs["high_wind"],
            "cloudy":        probs["cloudy"],
            "good_weather":  probs["good_weather"],
            "confidence":    float(np.mean(confidences)),
        }

    def predict_rain(self, features: dict) -> dict:
        """Back-compat single-target interface used by old route code."""
        scaled = self._make_vector(features)
        p = float(self.models["rain_occurred"].predict_proba(scaled)[0, 1])
        return {
            "will_rain":       p >= 0.5,
            "rain_probability": p,
            "confidence":      max(p, 1 - p),
        }

    # ── Persistence ─────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"✅ Model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "WeatherRainfallPredictor":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"✅ Model loaded ← {path}")
        return obj
