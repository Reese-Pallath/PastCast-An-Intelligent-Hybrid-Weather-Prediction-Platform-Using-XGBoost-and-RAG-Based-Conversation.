import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
import mlflow.sklearn
import pandas as pd

from ml.data_generator import generate_weather_dataset
from ml.predictor import WeatherRainfallPredictor, TARGETS

SCRIPT_DIR          = os.path.dirname(os.path.abspath(__file__))
DATA_PATH           = os.path.join(SCRIPT_DIR, "data",   "weather_data.csv")
MODEL_PATH          = os.path.join(SCRIPT_DIR, "models", "rain_predictor.pkl")
MLFLOW_TRACKING_URI = os.path.join(SCRIPT_DIR, "mlruns")
EXPERIMENT_NAME     = "PastCast-RainfallPrediction"


def _log_xgb_run(predictor, X_test, y_test_df):
    """Log all 5 XGBoost models + their metrics in a single MLflow run."""
    all_metrics = predictor.evaluate_all(X_test, y_test_df)

    with mlflow.start_run(run_name="XGBoost-MultiTarget"):
        mlflow.log_params({
            "n_estimators":     500,
            "max_depth":        5,
            "learning_rate":    0.05,
            "subsample":        0.85,
            "colsample_bytree": 0.80,
            "min_child_weight": 5,
            "calibration":      "isotonic",
            "n_samples":        len(y_test_df) * 5,   # approx total
            "n_targets":        5,
        })

        for col, label in TARGETS:
            m = all_metrics[col]
            prefix = col
            mlflow.log_metric(f"{prefix}_accuracy",  m["Accuracy"])
            mlflow.log_metric(f"{prefix}_roc_auc",   m["AUC-ROC"])
            mlflow.log_metric(f"{prefix}_f1",        m["F1-Score"])
            mlflow.log_metric(f"{prefix}_precision", m["Precision"])
            mlflow.log_metric(f"{prefix}_recall",    m["Recall"])

        # Log mean accuracy across all targets
        mean_acc = sum(m["Accuracy"] for m in all_metrics.values()) / len(all_metrics)
        mean_auc = sum(m["AUC-ROC"] for m in all_metrics.values()) / len(all_metrics)
        mlflow.log_metric("mean_accuracy", mean_acc)
        mlflow.log_metric("mean_roc_auc",  mean_auc)

        # Log each calibrated model as a sklearn artifact
        for col, label in TARGETS:
            mlflow.sklearn.log_model(
                sk_model=predictor.models[col],
                artifact_path=f"model_{col}",
                registered_model_name=f"PastCast-XGB-{col}",
            )

        run_id = mlflow.active_run().info.run_id
        print(f"\n  MLflow XGBoost run logged  run_id={run_id}")

    return all_metrics


def _log_rf_run(predictor, X_test, y_test_df):
    """Log Random Forest baseline (rain_occurred only)."""
    metrics = predictor.evaluate(
        predictor.baseline_model,
        X_test, y_test_df["rain_occurred"],
        label="Random Forest [Rain baseline]",
    )
    with mlflow.start_run(run_name="RandomForest-Baseline"):
        mlflow.log_params({"n_estimators": 300, "max_depth": 12,
                           "class_weight": "balanced"})
        mlflow.log_metric("accuracy",  metrics["Accuracy"])
        mlflow.log_metric("roc_auc",   metrics["AUC-ROC"])
        mlflow.log_metric("f1_score",  metrics["F1-Score"])
        mlflow.log_metric("precision", metrics["Precision"])
        mlflow.log_metric("recall",    metrics["Recall"])
        mlflow.sklearn.log_model(
            sk_model=predictor.baseline_model,
            artifact_path="model",
            registered_model_name="PastCast-RandomForest-Rain",
        )
        print(f"  MLflow RF run logged  run_id={mlflow.active_run().info.run_id}")
    return metrics


def main():
    print("=" * 65)
    print("  PastCast ML — Multi-Target Training Pipeline (MLflow)")
    print("=" * 65)

    mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_URI}")
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"\n  MLflow uri  : file://{MLFLOW_TRACKING_URI}")
    print(f"  Experiment  : {EXPERIMENT_NAME}\n")

    # ── Data ────────────────────────────────────────────────────
    if os.path.exists(DATA_PATH):
        print(f"Loading existing dataset: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        print(f"  Rows: {len(df)}")
        # Regenerate if it's the old 10k single-target dataset
        if len(df) < 40_000 or "extreme_heat" not in df.columns:
            print("  Dataset outdated — regenerating 50k multi-target dataset …")
            df = generate_weather_dataset(n_samples=50_000, output_path=DATA_PATH)
    else:
        print("Generating 50k synthetic dataset …")
        df = generate_weather_dataset(n_samples=50_000, output_path=DATA_PATH)

    predictor = WeatherRainfallPredictor()
    df = predictor.preprocess(df)

    # Print class balance for each target
    print("\nTarget class balance:")
    for col, label in TARGETS:
        rate = df[col].mean()
        print(f"  {label:16s}: {rate:.2%} positive")

    X_train, X_val, X_test, y_train, y_val, y_test = predictor.split_data(df)
    print(f"\nSplit  Train:{len(y_train)}  Val:{len(y_val)}  Test:{len(y_test)}")

    # ── XGBoost (all 5 targets) ─────────────────────────────────
    print("\nTraining XGBoost (5 targets) …")
    predictor.train_xgboost(X_train, X_val, y_train, y_val)
    xgb_metrics = _log_xgb_run(predictor, X_test, y_test)

    # Store rain metrics in performance_metrics for back-compat with route
    predictor.performance_metrics = xgb_metrics["rain_occurred"]

    # ── Random Forest baseline ───────────────────────────────────
    print("\nTraining Random Forest baseline (rain_occurred) …")
    predictor.train_random_forest(X_train, y_train)
    rf_metrics = _log_rf_run(predictor, X_test, y_test)
    predictor.baseline_metrics = rf_metrics

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(f"{'Target':<18} {'Accuracy':>10} {'AUC-ROC':>10} {'F1':>8}")
    print("-" * 50)
    for col, label in TARGETS:
        m = xgb_metrics[col]
        print(f"  {label:<16} {m['Accuracy']:>10.4f} {m['AUC-ROC']:>10.4f} {m['F1-Score']:>8.4f}")

    mean_acc = sum(xgb_metrics[c]["Accuracy"] for c, _ in TARGETS) / 5
    mean_auc = sum(xgb_metrics[c]["AUC-ROC"] for c, _ in TARGETS) / 5
    print("-" * 50)
    print(f"  {'MEAN':<16} {mean_acc:>10.4f} {mean_auc:>10.4f}")
    print(f"\n  RF baseline (rain): Acc={rf_metrics['Accuracy']:.4f}  AUC={rf_metrics['AUC-ROC']:.4f}")

    predictor.feature_importance(target_col="rain_occurred", top_n=8)
    predictor.save(MODEL_PATH)

    print(f"\nPipeline complete. Model saved → {MODEL_PATH}")
    print(f"\nView results:")
    print(f"  mlflow ui --backend-store-uri file://{MLFLOW_TRACKING_URI} --port 5001")
    print(f"  http://127.0.0.1:5001")


if __name__ == "__main__":
    main()
