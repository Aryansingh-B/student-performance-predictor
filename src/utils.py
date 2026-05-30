# src/utils.py
# ─────────────────────────────────────────────────────────────────
#  Reusable helper functions for preprocessing and evaluation
# ─────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (mean_absolute_error,
                              mean_squared_error, r2_score)


def load_data(path: str) -> pd.DataFrame:
    """Load dataset and drop helper columns if present."""
    df = pd.read_csv(path)
    df.drop(columns=['attendance_bin'], inplace=True, errors='ignore')
    return df


def encode_features(df: pd.DataFrame):
    """
    Encode categorical columns and return:
      - encoded DataFrame
      - dict of fitted LabelEncoders (needed for inference)
    """
    df = df.copy()
    encoders = {}

    cat_cols = ['parent_education_level', 'motivation_level']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders


def get_features_target(df: pd.DataFrame):
    """Split DataFrame into feature matrix X and target vector y."""
    X = df.drop(columns=['final_score'])
    y = df['final_score']
    return X, y


def evaluate_model(name: str, y_true, y_pred) -> dict:
    """Return a dict of regression metrics for one model."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    print(f"\n  ── {name} ──────────────────────────────")
    print(f"     MAE  : {mae:.4f}   (avg error in score points)")
    print(f"     RMSE : {rmse:.4f}   (penalises large errors more)")
    print(f"     R²   : {r2:.4f}   (1.0 = perfect fit)")

    return {'Model': name, 'MAE': round(mae,4),
            'RMSE': round(rmse,4), 'R2': round(r2,4)}