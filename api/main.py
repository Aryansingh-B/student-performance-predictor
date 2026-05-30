# api/main.py
# ─────────────────────────────────────────────────────────────────
#  FastAPI backend for Student Performance Predictor
#  Endpoints:
#    GET  /          → health check
#    GET  /info      → model info & metrics
#    POST /predict   → predict student score
#    GET  /docs      → Swagger UI (auto-generated)
# ─────────────────────────────────────────────────────────────────

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Literal
import joblib
import numpy as np
import pandas as pd
import os

# ── App Init ──────────────────────────────────────────────────────
app = FastAPI(
    title       = "Student Performance Predictor API",
    description = """
## 🎓 Student Performance Predictor

An ML-powered REST API that predicts a student's exam score
based on study habits, attendance, and personal factors.

**Built by:** Aryansingh Bais  
**Stack:** FastAPI · scikit-learn · pandas · joblib  
**Model:** Best model selected automatically from LR, DT, RF
    """,
    version     = "1.0.0",
    contact     = {
        "name" : "Aryansingh Bais",
        "url"  : "https://github.com/aryansinghbais",
    }
)

# ── CORS (allows Streamlit frontend to call this API) ─────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Load Model Artifact once at startup ───────────────────────────
MODEL_PATH = 'model/model.pkl'

def load_artifact():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run `python -m src.train` first."
        )
    return joblib.load(MODEL_PATH)

artifact = load_artifact()


# ════════════════════════════════════════════════════════════════
# ── Pydantic Schemas ──────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════

class StudentInput(BaseModel):
    """Input schema — all fields validated automatically by FastAPI."""

    study_hours_per_day: float = Field(
        ..., ge=0.5, le=12.0,
        description="Daily study hours (0.5 – 12)",
        example=6.0
    )
    attendance_percentage: float = Field(
        ..., ge=0.0, le=100.0,
        description="Class attendance percentage (0 – 100)",
        example=85.0
    )
    previous_score: float = Field(
        ..., ge=0.0, le=100.0,
        description="Score in previous exam (0 – 100)",
        example=72.0
    )
    sleep_hours_per_day: float = Field(
        ..., ge=3.0, le=12.0,
        description="Daily sleep hours (3 – 12)",
        example=7.0
    )
    extracurricular_activities: int = Field(
        ..., ge=0, le=1,
        description="Participates in extracurricular activities (0=No, 1=Yes)",
        example=1
    )
    parent_education_level: Literal[
        'No Formal Education', 'High School',
        'Graduate', 'Post-Graduate'
    ] = Field(
        ...,
        description="Highest education level of parent",
        example="Graduate"
    )
    internet_access: int = Field(
        ..., ge=0, le=1,
        description="Has internet access at home (0=No, 1=Yes)",
        example=1
    )
    tutoring_sessions_per_week: int = Field(
        ..., ge=0, le=10,
        description="Number of tutoring sessions per week (0 – 10)",
        example=2
    )
    motivation_level: Literal['Low', 'Medium', 'High'] = Field(
        ...,
        description="Student's self-reported motivation level",
        example="High"
    )

    # Extra safety validator
    @validator('study_hours_per_day')
    def study_hours_sanity(cls, v):
        if v > 12:
            raise ValueError("Study hours cannot exceed 24 hours in a day.")
        return v


class PredictionResponse(BaseModel):
    """Output schema returned by /predict endpoint."""
    predicted_score : float
    lower_bound     : float
    upper_bound     : float
    grade           : str
    model_used      : str
    confidence_band : str


class ModelInfo(BaseModel):
    """Schema for /info endpoint."""
    model_name    : str
    r2_score      : float
    mae           : float
    rmse          : float
    feature_count : int
    features      : list


# ════════════════════════════════════════════════════════════════
# ── Endpoints ─────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
def root():
    """Health check — confirms API is live."""
    return {
        "status"  : "✅ API is live",
        "message" : "Student Performance Predictor API v1.0.0",
        "docs"    : "/docs",
        "predict" : "/predict"
    }


@app.get("/info", response_model=ModelInfo, tags=["Model Info"])
def model_info():
    """Returns metadata about the loaded ML model."""
    return ModelInfo(
        model_name    = artifact['model_name'],
        r2_score      = round(artifact['metrics']['R2'],  4),
        mae           = round(artifact['metrics']['MAE'],  4),
        rmse          = round(artifact['metrics']['RMSE'], 4),
        feature_count = len(artifact['feature_names']),
        features      = artifact['feature_names']
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(student: StudentInput):
    """
    ## Predict Student Exam Score

    Send student details and receive:
    - **predicted_score** — estimated final exam score (0–100)
    - **lower_bound / upper_bound** — confidence band (± MAE)
    - **grade** — letter grade with label
    - **model_used** — which ML model made the prediction
    - **confidence_band** — human-readable range string
    """
    try:
        # Build input dict
        input_dict = student.dict()

        # Build DataFrame
        df = pd.DataFrame([input_dict])

        # Encode categoricals using saved encoders
        for col, le in artifact['encoders'].items():
            if col in df.columns:
                df[col] = le.transform(df[col])

        # Reorder columns to match training order
        df = df[artifact['feature_names']]

        # Predict
        raw_score = float(artifact['model'].predict(df)[0])
        score     = round(float(np.clip(raw_score, 0, 100)), 1)

        # Confidence band
        mae   = artifact['metrics'].get('MAE', 3.5)
        lower = round(max(0,   score - mae), 1)
        upper = round(min(100, score + mae), 1)

        # Grade logic
        if   score >= 85: grade = "A+ — Excellent"
        elif score >= 75: grade = "A  — Very Good"
        elif score >= 65: grade = "B  — Good"
        elif score >= 55: grade = "C  — Average"
        elif score >= 45: grade = "D  — Below Average"
        else:             grade = "F  — Needs Improvement"

        return PredictionResponse(
            predicted_score = score,
            lower_bound     = lower,
            upper_bound     = upper,
            grade           = grade,
            model_used      = artifact['model_name'],
            confidence_band = f"{lower} – {upper}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))