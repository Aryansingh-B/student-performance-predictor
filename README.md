# 🎓 Student Performance Predictor

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> An end-to-end Machine Learning web application that predicts a student's
> exam score based on study habits, attendance, sleep, motivation, and family
> background — built with a full data science pipeline from EDA to deployment.

**Built by:** Aryansingh Bais — Aspiring Data Scientist
**Training:** Data Science with GenAI & Agentic AI — NareshIT, Hyderabad
**Connect:** [LinkedIn](https://linkedin.com/in/aryansinghbais) · [GitHub](https://github.com/aryansinghbais)

---

## 🚀 Live Demo

> 🔗 **[Click here to try the live app →](https://your-app-name.streamlit.app)**
> *(Deployed on Streamlit Cloud — free, no login needed)*

---

## 📌 What This Project Does

This project answers one real-world question:

> **"Given a student's daily habits and background, what exam score
> will they likely get — and what grade does that map to?"**

A user inputs **9 features** about a student and the app returns:
- A **predicted exam score** (0–100)
- A **confidence band** — score ± MAE, e.g. `65.3 – 72.3`
- A **letter grade** (A+ to F) with a performance label
- Which **ML model** made the prediction and its accuracy

### Why this project?
Most student performance problems in real companies (edtech, schools,
coaching institutes) involve exactly this kind of tabular ML pipeline.
This project demonstrates the complete workflow a Data Scientist follows:
data creation → EDA → feature engineering → model training → API →
frontend → containerization → deployment.

---


## 🛠️ Tech Stack
- **EDA & ML:** pandas, numpy, scikit-learn, matplotlib, seaborn
- **Backend API:** FastAPI
- **Frontend:** Streamlit
- **Containerization:** Docker
- **Deployment:** Streamlit Cloud

## 📁 Project Structure

```
student-performance-predictor/
│
├── data/
│   ├── generate_data.py     ← Creates 1000-row synthetic dataset
│   ├── student_data.csv     ← Generated dataset
│   └── plot_*.png           ← 15+ EDA & model evaluation plots
│
├── notebooks/
│   └── 01_EDA_and_Model.ipynb  ← Full EDA: correlations, outliers,
│                                  feature importance, distributions
├── src/
│   ├── utils.py             ← Shared helpers: load, encode, evaluate
│   ├── train.py             ← Trains 3 models, picks best, saves .pkl
│   └── predict.py           ← Inference: loads artifact, returns score
│
├── app/
│   └── streamlit_app.py     ← Frontend: sliders, gauge chart, radar,
│                               EDA tabs, confidence band, smart tips
├── api/
│   └── main.py              ← FastAPI: /predict /info /health + Swagger
│
├── model/                   ← model.pkl saved here (gitignored)
├── Dockerfile               ← Streamlit container
├── Dockerfile.api           ← FastAPI container
├── docker-compose.yml       ← Runs both services together
└── requirements.txt         ← All dependencies pinned
```

## ⚙️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/aryansinghbais/student-performance-predictor.git
cd student-performance-predictor

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model
python src/train.py

# 5. Run Streamlit app
streamlit run app/streamlit_app.py
```

## 👤 Author
**Aryansingh Bais** — [LinkedIn](https://www.linkedin.com/in/aryansinghbais8/) · [GitHub](https://github.com/Aryansingh-B)