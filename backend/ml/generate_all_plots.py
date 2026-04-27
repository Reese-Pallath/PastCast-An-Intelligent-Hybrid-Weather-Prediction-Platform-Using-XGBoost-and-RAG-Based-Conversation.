import os
import sys
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predictor import WeatherRainfallPredictor

def generate_plots():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'models', 'rain_predictor.pkl')
    data_path = os.path.join(script_dir, 'data', 'weather_data.csv')
    
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        predictor = pickle.load(f)
        
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df = predictor.preprocess(df)
    X_train, X_val, X_test, y_train, y_val, y_test = predictor.split_data(df)
    
    # Setup plot style
    sns.set_theme(style="whitegrid")
    
    # 1. Confusion Matrix
    print("Generating Confusion Matrix...")
    y_pred = predictor.models['rain_occurred'].predict(X_test)
    cm = confusion_matrix(y_test['rain_occurred'], y_pred)
    
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Rain Prediction', fontsize=16, pad=20, fontweight='bold')
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    out_cm = os.path.join(script_dir, '..', 'confusion_matrix.png')
    plt.savefig(out_cm, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_cm}")
    
    # 2. ROC-AUC Curve
    print("Generating ROC-AUC Curve...")
    y_prob = predictor.models['rain_occurred'].predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test['rain_occurred'], y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=16, pad=20, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    out_roc = os.path.join(script_dir, '..', 'roc_auc_curve.png')
    plt.savefig(out_roc, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_roc}")
    
    # 3. Model Accuracy Comparison
    print("Generating Model Accuracy Comparison...")
    metrics = predictor.performance_metrics if hasattr(predictor, 'performance_metrics') else {}
    baseline = predictor.baseline_metrics if hasattr(predictor, 'baseline_metrics') else {}
    
    if not metrics or not baseline:
        print("Performance metrics not found in model object, evaluating now...")
        metrics = predictor.evaluate(predictor.models['rain_occurred'], X_test, y_test['rain_occurred'], "XGBoost")
        baseline = predictor.evaluate(predictor.baseline_model, X_test, y_test['rain_occurred'], "Random Forest")
        
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    xgb_scores = [metrics.get(l, 0) for l in labels]
    rf_scores = [baseline.get(l, 0) for l in labels]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10,6))
    rects1 = ax.bar(x - width/2, xgb_scores, width, label='XGBoost', color='royalblue')
    rects2 = ax.bar(x + width/2, rf_scores, width, label='Random Forest', color='lightcoral')
    
    ax.set_ylabel('Scores', fontsize=14)
    ax.set_title('Model Performance Comparison', fontsize=16, pad=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim([0, 1.1])
    
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
                        
    out_cmp = os.path.join(script_dir, '..', 'model_accuracy_comparison.png')
    plt.savefig(out_cmp, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_cmp}")

if __name__ == "__main__":
    generate_plots()
