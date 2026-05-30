# 🎓 Student Performance Predictor

An end-to-end Machine Learning web application that predicts a student's
exam score based on study habits, attendance, and past performance.

## 🚀 Demo
> _Live demo link coming soon — deploying on Streamlit Cloud_

## 🛠️ Tech Stack
- **EDA & ML:** pandas, numpy, scikit-learn, matplotlib, seaborn
- **Backend API:** FastAPI
- **Frontend:** Streamlit
- **Containerization:** Docker
- **Deployment:** Streamlit Cloud

## 📁 Project Structure
\`\`\`
student-performance-predictor/
├── data/               → Dataset
├── notebooks/          → EDA & Model training notebook
├── src/                → Training & prediction scripts
├── app/                → Streamlit frontend
├── api/                → FastAPI backend
├── model/              → Saved ML model
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
\`\`\`

## ⚙️ Run Locally

\`\`\`bash
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
\`\`\`

## 👤 Author
**Aryansingh Bais** — [LinkedIn](https://www.linkedin.com/in/aryansinghbais8/) · [GitHub](https://github.com/Aryansingh-B)