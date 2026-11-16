# 🎓 Student Performance Prediction System

> **Turning Student Data into Actionable Academic Insights**
>
> An end-to-end machine learning application that predicts student exam scores using academic, lifestyle, and background factors. Built with **Python, Scikit-learn, FastAPI, and Streamlit**, the project compares multiple regression models and delivers predictions through an interactive web application. The best evaluated model, Random Forest, achieved an **R² score of 0.85** with an **MAE of 4.2 points** on a 0–100 exam score scale.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit--learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**Quick Links:** [Live Demo](https://student-performance-predictor-hciqulnwu7e7n5fjfsaz9b.streamlit.app/) · [Dataset](#dataset-overview) · [Model Results](#key-insights--findings) · [Project Structure](#repository-structure--how-to-run)

---

## 1. Business Problem / Problem Statement

Educational institutions need ways to identify students who may be at risk of poor academic performance before final examinations.

Without data-driven insights, educators may struggle to identify the factors influencing student outcomes, prioritize academic support, and allocate tutoring resources effectively.

**Business objective:** Develop a predictive system that estimates exam performance and helps educators identify students who may benefit from timely academic intervention.

## 2. Dataset Overview & Data Dictionary

The project uses a **synthetically generated dataset of 1,000 student records** for model development and evaluation.

| Field | Data Type | Business Definition |
|---|---|---|
| `study_hours` | Float | Average daily study hours |
| `attendance` | Numeric | Student attendance percentage |
| `sleep_hours` | Float | Average daily sleep duration |
| `motivation` | Numeric | Student motivation level |
| `previous_gpa` | Float | Previous academic performance |
| `family_income` | Numeric | Family income |
| `parental_education` | Categorical | Parent's education level |
| `school_type` | Categorical | Public or private school |
| `grade_level` | Integer | Student's current grade |
| `exam_score` | Numeric | Target variable: predicted exam score (0–100) |

**Dataset details:**
- Records: **1,000**
- Input features: **9**
- Target variable: `exam_score`
- Source: Synthetically generated for educational purposes.
- Generation script: `data/generate_data.py`

## 3. Project Objectives & Key Business Questions

### Analytical Objectives
- Develop a regression model to estimate student exam scores.
- Identify the factors most associated with academic performance.
- Compare multiple machine learning models using evaluation metrics.
- Build an interactive application for individual student predictions and data exploration.

### Key Business Questions
1. How accurately can exam scores be predicted using student-related factors?
2. Which factors have the greatest influence on predicted performance?
3. How does study time relate to academic outcomes?
4. What role do attendance and sleep patterns play in student performance?
5. Which regression model provides the most reliable predictions?
6. How can prediction ranges help educators interpret model uncertainty?

## 4. Methodology & Analytical Workflow

| Stage | Work Performed | Tools |
|---|---|---|
| Data Generation | Created a synthetic dataset with 1,000 student records | Python, Pandas |
| Data Preparation | Prepared numerical and categorical features for modeling | Pandas, NumPy |
| EDA | Explored distributions, correlations, and performance patterns through 10+ visualizations | Matplotlib, Seaborn, Plotly |
| Model Development | Trained and compared three regression algorithms | Scikit-learn |
| Model Evaluation | Evaluated models using R², MAE, RMSE, and 5-fold cross-validation | Scikit-learn |
| API Development | Created prediction, batch prediction, model information, and health-check endpoints | FastAPI |
| Application Development | Built an interactive prediction interface and exploration dashboard | Streamlit |
| Deployment | Containerized the application and API for local execution | Docker, Docker Compose |

### Model Comparison

| Model | R² Score | MAE | RMSE |
|---|---:|---:|---:|
| **Random Forest** | **0.85** | **4.2** | 5.8 |
| Decision Tree | 0.80 | 5.1 | 7.2 |
| Linear Regression | 0.78 | 5.8 | 8.1 |

## 5. Key Insights & Findings

- **Model performance:** Random Forest achieved the best test performance with an **R² of 0.85** and **MAE of 4.2 points**.
- **Prediction consistency:** Five-fold cross-validation produced a mean R² of **0.85 ± 0.02**, indicating consistent performance across folds.
- **Study habits:** Study hours per day ranked as the strongest predictor in the reported feature-importance analysis.
- **Student motivation:** Motivation level was identified among the top five predictive features.
- **Academic history:** Previous academic performance was also a key predictor of exam scores.
- **Prediction uncertainty:** The application displays a confidence band using the model's MAE to help users interpret estimated scores.

## 6. Actionable Business Recommendations

| Finding | Recommendation | Potential Business Value |
|---|---|---|
| Study hours are a leading predictor | Identify students with limited study time and offer structured study plans | Support more focused academic interventions |
| Motivation is an important feature | Introduce mentoring and regular progress check-ins | Help improve student engagement |
| Previous performance influences predictions | Use previous academic results to identify students needing additional support | Enable earlier intervention planning |
| Predictions include an error range | Use prediction ranges alongside teacher assessments rather than relying on a single score | Support more informed decisions |
| The API supports batch predictions | Explore batch scoring for larger student groups | Reduce manual assessment effort |

*These are proposed applications of the model, not measured improvements in student outcomes. Predictions should support—not replace—educator judgment.*

## 7. Repository Structure & How to Run

```text
student-performance-predictor/
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── generate_data.py
│   └── student_data.csv
│
├── model/
│   └── model.pkl
│
├── notebooks/
│   └── 01_EDA_and_model.ipynb
│
├── src/
│   ├── __init__.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
│
├── .streamlit/
├── Dockerfile
├── Dockerfile.api
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
└── README.md
