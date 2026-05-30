# src/train.py
# ─────────────────────────────────────────────────────────────────
#  Trains 3 ML models, compares them, saves the best one.
#  Run: python src/train.py
# ─────────────────────────────────────────────────────────────────

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model    import LinearRegression
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor
from sklearn.preprocessing   import StandardScaler

from src.utils import (load_data, encode_features,
                        get_features_target, evaluate_model)

# ── Config ────────────────────────────────────────────────────────
DATA_PATH  = 'data/student_data.csv'
MODEL_DIR  = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
TEST_SIZE  = 0.20
RANDOM_STATE = 42

sns.set_theme(style='whitegrid', font_scale=1.05)
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False,
                      "axes.spines.right": False})

os.makedirs(MODEL_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════
def main():

    # ── 1. Load & Preprocess ──────────────────────────────────────
    print("\n🔄 Loading data...")
    df = load_data(DATA_PATH)
    df_enc, encoders = encode_features(df)
    X, y = get_features_target(df_enc)
    feature_names = X.columns.tolist()

    print(f"   Features : {X.shape[1]}  |  Samples : {X.shape[0]}")

    # ── 2. Train / Test Split ─────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"   Train : {X_train.shape[0]} rows  |  Test : {X_test.shape[0]} rows")

    # Scale features (for Linear Regression)
    scaler  = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── 3. Define Models ──────────────────────────────────────────
    models = {
        'Linear Regression' : LinearRegression(),
        'Decision Tree'     : DecisionTreeRegressor(
                                max_depth=8,
                                min_samples_split=10,
                                random_state=RANDOM_STATE),
        'Random Forest'     : RandomForestRegressor(
                                n_estimators=200,
                                max_depth=10,
                                min_samples_split=5,
                                random_state=RANDOM_STATE,
                                n_jobs=-1),
    }

    # ── 4. Train & Evaluate ───────────────────────────────────────
    print("\n📊 Training & Evaluating Models...")
    print("=" * 55)

    results   = []
    trained   = {}
    cv_scores = {}

    for name, model in models.items():
        # Use scaled data for Linear Regression only
        Xtr = X_train_sc if name == 'Linear Regression' else X_train
        Xte = X_test_sc  if name == 'Linear Regression' else X_test

        # Fit
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)

        # Metrics
        metrics = evaluate_model(name, y_test, y_pred)
        results.append(metrics)
        trained[name] = (model, y_pred)

        # 5-fold Cross Validation
        cv = cross_val_score(model, Xtr, y_train,
                             cv=5, scoring='r2', n_jobs=-1)
        cv_scores[name] = cv
        print(f"     CV R² : {cv.mean():.4f} ± {cv.std():.4f}")

    # ── 5. Results Table ──────────────────────────────────────────
    results_df = pd.DataFrame(results).set_index('Model')
    print(f"\n{'='*55}")
    print("  MODEL COMPARISON TABLE")
    print(f"{'='*55}")
    print(results_df.to_string())

    # ── 6. Pick Best Model ────────────────────────────────────────
    best_name = results_df['R2'].idxmax()
    best_r2   = results_df.loc[best_name, 'R2']
    print(f"\n🏆 Best Model : {best_name}  (R² = {best_r2})")

    # ── 7. Plot 1 — Model Comparison Bar Chart ────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics_to_plot = ['R2', 'MAE', 'RMSE']
    colors = ['#42A5F5', '#EF5350', '#AB47BC']

    for ax, metric, color in zip(axes, metrics_to_plot, colors):
        vals  = results_df[metric]
        bars  = ax.bar(vals.index, vals.values, color=color,
                       edgecolor='white', width=0.5)
        for bar, val in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=10)
        ax.set_title(f'{metric} Comparison', fontweight='bold')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=15)

    plt.suptitle("Model Performance Comparison",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('data/plot_model_comparison.png', bbox_inches='tight')
    plt.show()
    print("   💾 Saved → data/plot_model_comparison.png")

    # ── 8. Plot 2 — Actual vs Predicted (Best Model) ──────────────
    best_model, best_pred = trained[best_name]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Scatter: actual vs predicted
    axes[0].scatter(y_test, best_pred, alpha=0.5,
                    color='steelblue', edgecolors='white',
                    linewidth=0.3, s=40)
    lims = [min(y_test.min(), best_pred.min()) - 2,
            max(y_test.max(), best_pred.max()) + 2]
    axes[0].plot(lims, lims, 'r--', linewidth=1.5, label='Perfect Prediction')
    axes[0].set_title(f'{best_name} — Actual vs Predicted',
                      fontweight='bold')
    axes[0].set_xlabel('Actual Score')
    axes[0].set_ylabel('Predicted Score')
    axes[0].legend()

    # Residuals
    residuals = y_test.values - best_pred
    axes[1].hist(residuals, bins=30, color='#66BB6A',
                 edgecolor='white', linewidth=0.5)
    axes[1].axvline(0, color='red', linestyle='--',
                    linewidth=1.5, label='Zero Error')
    axes[1].set_title('Residual Distribution', fontweight='bold')
    axes[1].set_xlabel('Residual (Actual − Predicted)')
    axes[1].set_ylabel('Count')
    axes[1].legend()

    plt.suptitle(f"Best Model Analysis — {best_name}",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('data/plot_actual_vs_predicted.png', bbox_inches='tight')
    plt.show()
    print("   💾 Saved → data/plot_actual_vs_predicted.png")

    # ── 9. Plot 3 — Feature Importance (Random Forest) ───────────
    rf_model = trained['Random Forest'][0]
    importances = pd.Series(
        rf_model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    bar_colors = ['#1565C0' if v > importances.median()
                  else '#90CAF9' for v in importances.values]
    bars = ax.barh(importances.index, importances.values,
                   color=bar_colors, edgecolor='white')

    for bar, val in zip(bars, importances.values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    ax.set_title("Random Forest — Feature Importances",
                 fontsize=13, fontweight='bold')
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig('data/plot_feature_importances_rf.png', bbox_inches='tight')
    plt.show()
    print("   💾 Saved → data/plot_feature_importances_rf.png")

    # ── 10. Cross Validation Plot ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    positions = range(len(cv_scores))
    for pos, (name, scores) in zip(positions, cv_scores.items()):
        ax.plot([pos]*5, scores, 'o', alpha=0.6, markersize=7)
        ax.plot(pos, scores.mean(), 'D', markersize=11,
                label=f'{name} (mean={scores.mean():.3f})')

    ax.set_xticks(list(positions))
    ax.set_xticklabels(cv_scores.keys(), rotation=10)
    ax.set_ylabel("CV R² Score")
    ax.set_title("5-Fold Cross Validation R² Scores",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig('data/plot_cross_validation.png', bbox_inches='tight')
    plt.show()
    print("   💾 Saved → data/plot_cross_validation.png")

    # ── 11. Save Best Model ───────────────────────────────────────
    # Dynamically save whichever model actually performed best
    best_model_obj = trained[best_name][0]

    artifact = {
        'model'         : best_model_obj,
        'scaler'        : scaler,
        'encoders'      : encoders,
        'feature_names' : feature_names,
        'model_name'    : 'Random Forest',
        'metrics'       : results_df.loc['Random Forest'].to_dict()
    }

    joblib.dump(artifact, MODEL_PATH)
    print(f"\n✅ Model artifact saved → {MODEL_PATH}")
    print(f"   Contains: model + scaler + encoders + feature_names + metrics")

    # ── 12. Final Summary ─────────────────────────────────────────
    print(f"""
╔══════════════════════════════════════════════════════╗
║           TRAINING COMPLETE — SUMMARY               ║
╠══════════════════════════════════════════════════════╣
║  Best Model  : {best_name:<37}║
║  R² Score    : {results_df.loc[best_name,'R2']:.4f}                              ║
║  MAE         : {results_df.loc[best_name,'MAE']:.4f} points                        ║
║  RMSE        : {results_df.loc[best_name,'RMSE']:.4f} points                        ║
║  'model_name'    : best_name,                      ║
║  'metrics'       : results_df.loc[best_name].to_dict()           ║
╚══════════════════════════════════════════════════════╝
""")


# ════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    main()