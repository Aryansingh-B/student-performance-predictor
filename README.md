# 🎓 Student Performance Predictor

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?logo=render&logoColor=white)
![Streamlit Cloud](https://img.shields.io/badge/Streamlit_Cloud-Deployed-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> An end-to-end machine learning web application that predicts a student's exam score based on study habits, attendance, motivation, and background factors.

---

## 🚀 Live Demo

| Service | URL |
|---|---|
| 🎯 **Streamlit App** | [Coming Soon — deploying on Streamlit Cloud] |
| ⚡ **FastAPI Swagger** | [student-performance-predictor-api-cf6d.onrender.com/docs](https://student-performance-predictor-api-cf6d.onrender.com/docs) |

---

## 📸 Preview

> Predicts exam scores (0–100) with grade classification (A+, A, B, C, D, F) and confidence band using a Random Forest model with R² = 0.85 and MAE ± 4.2.

---

## 🧠 Features

- **ML Pipeline** — Data generation → EDA → Feature engineering → Model training → Evaluation
- **3 Models Compared** — Linear Regression, Decision Tree, Random Forest (best selected automatically)
- **REST API** — FastAPI backend with Pydantic validation and Swagger UI
- **Interactive UI** — Streamlit frontend with score gauge, grade badge, and personalized feedback
- **Containerized** — Full Docker + docker-compose setup for one-command local deployment
- **Auto-training** — Model trains automatically on first run if `model.pkl` doesn't exist

---

## 🏗️ Project Structure

```
student-performance-predictor/
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI app with /predict and /health endpoints
├── app/
│   └── streamlit_app.py     # Streamlit frontend
├── data/
│   ├── generate_data.py     # Synthetic dataset generator (1000 students)
│   └── student_data.csv     # Generated dataset
├── src/
│   └── train.py             # Model training, evaluation, and plots
├── model/                   # Saved model.pkl (gitignored, auto-generated)
├── notebooks/               # EDA and exploration notebooks
├── Dockerfile               # Streamlit app container
├── Dockerfile.api           # FastAPI container
├── docker-compose.yml       # Orchestrates both services
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Development + notebook dependencies
└── startup.sh               # Auto-train script for cloud deployment
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| ML | Scikit-learn, NumPy, Pandas |
| API | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit, Plotly |
| Containerization | Docker, Docker Compose |
| API Deployment | Render |
| App Deployment | Streamlit Cloud |

---

## 🔧 Run Locally

### Method 1 — Without Docker (Recommended for development)

```bash
# 1. Clone the repo
git clone https://github.com/Aryansingh-B/student-performance-predictor.git
cd student-performance-predictor

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements-dev.txt

# 4. Generate dataset
python data/generate_data.py

# 5. Train the model
python -m src.train

# 6. Run Streamlit app
streamlit run app/streamlit_app.py

# 7. Run FastAPI (separate terminal)
uvicorn api.main:app --reload --port 8000
```

- Streamlit → http://localhost:8501
- FastAPI Swagger → http://localhost:8000/docs

### Method 2 — Docker (One command)

```bash
docker-compose up --build
```

- Streamlit → http://localhost:8501
- FastAPI Swagger → http://localhost:8000/docs

---

## 📊 Model Performance

| Model | R² Score | MAE |
|---|---|---|
| Linear Regression | ~0.78 | ~5.8 |
| Decision Tree | ~0.80 | ~5.1 |
| **Random Forest** ✅ | **0.85** | **±4.2** |

Random Forest was selected automatically as the best performing model.

---

## 🎯 Input Features

| Feature | Type | Range |
|---|---|---|
| Study Hours per Day | Numeric | 1 – 10 |
| Attendance Percentage | Numeric | 50 – 100% |
| Previous Score | Numeric | 30 – 95 |
| Sleep Hours per Day | Numeric | 4 – 10 |
| Tutoring Sessions per Week | Numeric | 0 – 5 |
| Motivation Level | Categorical | Low / Medium / High |
| Parent Education Level | Categorical | None / High School / Graduate / Post-Graduate |
| Extracurricular Activities | Binary | Yes / No |
| Internet Access at Home | Binary | Yes / No |

---

## 📡 API Usage

```bash
curl -X POST "https://student-performance-predictor-api-cf6d.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "study_hours_per_day": 7,
    "attendance_percentage": 85,
    "previous_score": 75,
    "sleep_hours_per_day": 7,
    "tutoring_sessions_per_week": 2,
    "motivation_level": "High",
    "parent_education_level": "Graduate",
    "extracurricular_activities": 1,
    "internet_access": 1
  }'
```

**Response:**
```json
{
  "predicted_score": 84.5,
  "grade": "A",
  "confidence_low": 80.3,
  "confidence_high": 88.7
}
```

---

## 👨‍💻 Author

**Aryansingh Bais**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/aryansingh-bais)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/Aryansingh-B)

---

## 📄 License

This project is licensed under the MIT License.