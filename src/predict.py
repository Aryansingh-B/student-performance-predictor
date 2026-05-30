# src/predict.py
# ─────────────────────────────────────────────────────────────────
#  Loads saved model artifact and runs inference on new input.
#  Used by both Streamlit app and FastAPI backend.
# ─────────────────────────────────────────────────────────────────

import joblib
import pandas as pd
import numpy as np

MODEL_PATH = 'model/model.pkl'

def load_artifact():
    """Load the saved model artifact (model + encoders + scaler)."""
    return joblib.load(MODEL_PATH)


def predict_score(input_dict: dict) -> dict:
    """
    Takes raw user input as a dict, preprocesses it,
    and returns predicted score + confidence band.

    Parameters
    ----------
    input_dict : dict
        Keys must match original feature names exactly.

    Returns
    -------
    dict with keys: predicted_score, lower_bound, upper_bound, grade
    """
    artifact = load_artifact()
    model         = artifact['model']
    encoders      = artifact['encoders']
    feature_names = artifact['feature_names']

    # Build DataFrame from input
    df = pd.DataFrame([input_dict])

    # Encode categorical columns using saved encoders
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])

    # Reorder columns to match training
    df = df[feature_names]

    # Predict
    score = float(np.clip(model.predict(df)[0], 0, 100))

    # Confidence band (±MAE from training)
    mae = artifact['metrics'].get('MAE', 3.5)
    lower = round(max(0,   score - mae), 1)
    upper = round(min(100, score + mae), 1)

    # Grade
    if score >= 85:   grade = "A+ — Excellent"
    elif score >= 75: grade = "A  — Very Good"
    elif score >= 65: grade = "B  — Good"
    elif score >= 55: grade = "C  — Average"
    elif score >= 45: grade = "D  — Below Average"
    else:             grade = "F  — Needs Improvement"

    return {
        'predicted_score': round(score, 1),
        'lower_bound'    : lower,
        'upper_bound'    : upper,
        'grade'          : grade
    }


# ── Quick CLI test ────────────────────────────────────────────────
if __name__ == '__main__':
    sample = {
        'study_hours_per_day'       : 6.0,
        'attendance_percentage'     : 85.0,
        'previous_score'            : 72.0,
        'sleep_hours_per_day'       : 7.0,
        'extracurricular_activities': 1,
        'parent_education_level'    : 'Graduate',
        'internet_access'           : 1,
        'tutoring_sessions_per_week': 2,
        'motivation_level'          : 'High'
    }
    result = predict_score(sample)
    print("\n── Prediction Result ──────────────────────────────")
    for k, v in result.items():
        print(f"   {k:<20} : {v}")