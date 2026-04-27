import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def generate_importance_chart():

    model_path = os.path.join(os.path.dirname(__file__), 'models', 'rain_predictor.pkl')

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Loading trained XGBoost model from {model_path}...")
    with open(model_path, 'rb') as f:
        predictor = pickle.load(f)

    if hasattr(predictor.model, 'feature_importances_'):
        importances = predictor.model.feature_importances_
    else:
        print("Model doesn't expose feature_importances_. Make sure it is trained.")
        return

    features = predictor.feature_names

    indices = np.argsort(importances)[::-1]

    top_n = min(10, len(features))
    top_indices = indices[:top_n]
    top_features = [features[i] for i in top_indices]
    top_importances = importances[top_indices]

    print("\n--- TOP PREDICTIVE FEATURES ---")
    for i in range(top_n):
        print(f"{i+1}. {top_features[i]}: {top_importances[i]:.4f}")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    colors = sns.color_palette("viridis", top_n)

    bars = plt.bar(range(top_n), top_importances, color=colors)
    plt.title('Feature Importance: What drives rainfall predictions?', fontsize=16, pad=20, fontweight='bold')

    plt.xticks(range(top_n), top_features, rotation=35, ha='right', fontsize=12)
    plt.yticks(fontsize=11)
    plt.ylabel('Relative Importance (Contribution Score)', fontsize=14, labelpad=10)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'feature_importance_chart.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ High-resolution chart saved to: {output_path}")

if __name__ == "__main__":
    generate_importance_chart()
