# app/streamlit_app.py
# ─────────────────────────────────────────────────────────────────
#  Streamlit frontend for Student Performance Predictor
#  Run: streamlit run app/streamlit_app.py
# ─────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title = "Student Performance Predictor",
    page_icon  = "🎓",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f8fafc; }

    /* Prediction result card */
    .result-card {
        background: linear-gradient(135deg, #1A3A5C 0%, #2E75B6 100%);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(26,58,92,0.3);
    }
    .result-score {
        font-size: 4.5rem;
        font-weight: 800;
        letter-spacing: -2px;
        margin: 0;
    }
    .result-grade {
        font-size: 1.4rem;
        opacity: 0.92;
        margin-top: 0.3rem;
    }
    .result-band {
        font-size: 1rem;
        opacity: 0.75;
        margin-top: 0.4rem;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        border-left: 4px solid #2E75B6;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1A3A5C;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #666;
        margin-top: 0.2rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1A3A5C;
        border-bottom: 2px solid #2E75B6;
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
    }

    /* Tip box */
    .tip-box {
        background: #EFF6FF;
        border-left: 4px solid #2E75B6;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        color: #1A3A5C;
        margin: 0.5rem 0;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# ── Load Model ───────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    """Load model artifact once and cache it."""
    path = 'model/model.pkl'
    if not os.path.exists(path):
        return None
    return joblib.load(path)

artifact = load_model()


# ════════════════════════════════════════════════════════════════
# ── Helper Functions ─────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════

def get_grade_color(score):
    if score >= 85: return "#22c55e"   # green
    elif score >= 75: return "#84cc16" # lime
    elif score >= 65: return "#f59e0b" # amber
    elif score >= 55: return "#f97316" # orange
    elif score >= 45: return "#ef4444" # red
    else: return "#dc2626"             # dark red

def get_grade(score):
    if score >= 85:   return "A+ — Excellent"
    elif score >= 75: return "A  — Very Good"
    elif score >= 65: return "B  — Good"
    elif score >= 55: return "C  — Average"
    elif score >= 45: return "D  — Below Average"
    else:             return "F  — Needs Improvement"

def predict(input_dict):
    """Run inference using loaded artifact."""
    df = pd.DataFrame([input_dict])
    for col, le in artifact['encoders'].items():
        if col in df.columns:
            df[col] = le.transform(df[col])
    df = df[artifact['feature_names']]
    score = float(np.clip(artifact['model'].predict(df)[0], 0, 100))
    mae   = artifact['metrics'].get('MAE', 3.5)
    return {
        'score' : round(score, 1),
        'lower' : round(max(0,   score - mae), 1),
        'upper' : round(min(100, score + mae), 1),
        'grade' : get_grade(score),
        'color' : get_grade_color(score)
    }

def make_gauge(score, color):
    """Plotly gauge chart for predicted score."""
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = score,
        domain = {'x': [0,1], 'y': [0,1]},
        title  = {'text': "Predicted Score", 'font': {'size': 16}},
        number = {'suffix': "/100", 'font': {'size': 36, 'color': color}},
        gauge  = {
            'axis'  : {'range': [0, 100], 'tickwidth': 1},
            'bar'   : {'color': color, 'thickness': 0.3},
            'steps' : [
                {'range': [0,  45], 'color': '#fee2e2'},
                {'range': [45, 55], 'color': '#ffedd5'},
                {'range': [55, 65], 'color': '#fef9c3'},
                {'range': [65, 75], 'color': '#dcfce7'},
                {'range': [75, 85], 'color': '#bbf7d0'},
                {'range': [85,100], 'color': '#86efac'},
            ],
            'threshold': {
                'line' : {'color': color, 'width': 4},
                'thickness': 0.85,
                'value': score
            }
        }
    ))
    fig.update_layout(
        height=260, margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def make_radar(input_dict):
    """Radar chart showing student profile vs ideal student."""
    categories = [
        'Study Hours', 'Attendance', 'Prev Score',
        'Sleep', 'Tutoring', 'Motivation'
    ]
    # Normalize each to 0–10 scale
    motivation_map = {'Low': 2, 'Medium': 5, 'High': 9}
    student_vals = [
        input_dict['study_hours_per_day'],                       # already 1–10
        input_dict['attendance_percentage'] / 10,                # 0–100 → 0–10
        input_dict['previous_score'] / 10,                       # 0–100 → 0–10
        input_dict['sleep_hours_per_day'],                       # already 3–10
        input_dict['tutoring_sessions_per_week'] * 2,            # 0–5 → 0–10
        motivation_map[input_dict['motivation_level']]           # mapped to 0–10
    ]
    ideal_vals = [9, 9.5, 8.5, 8, 8, 9]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=student_vals + [student_vals[0]],
        theta=categories + [categories[0]],
        fill='toself', name='This Student',
        line_color='#2E75B6', fillcolor='rgba(46,117,182,0.2)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=ideal_vals + [ideal_vals[0]],
        theta=categories + [categories[0]],
        fill='toself', name='Ideal Student',
        line_color='#22c55e', fillcolor='rgba(34,197,94,0.1)',
        line_dash='dash'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=True, height=320,
        margin=dict(t=30, b=30, l=40, r=40),
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=-0.2)
    )
    return fig

def make_feature_bar(input_dict):
    """Horizontal bar showing how each feature compares to average."""
    dataset = pd.read_csv('data/student_data.csv')
    avg = dataset[[
        'study_hours_per_day','attendance_percentage',
        'previous_score','sleep_hours_per_day',
        'tutoring_sessions_per_week'
    ]].mean()

    features = list(avg.index)
    student_v = [input_dict[f] for f in features]
    avg_v     = avg.values.tolist()
    labels    = [f.replace('_',' ').title() for f in features]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='This Student', x=student_v, y=labels,
        orientation='h', marker_color='#2E75B6',
        text=[f'{v:.1f}' for v in student_v],
        textposition='outside'
    ))
    fig.add_trace(go.Bar(
        name='Dataset Average', x=avg_v, y=labels,
        orientation='h', marker_color='#94a3b8',
        text=[f'{v:.1f}' for v in avg_v],
        textposition='outside'
    ))
    fig.update_layout(
        barmode='group', height=300,
        margin=dict(t=20, b=20, l=10, r=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=-0.35),
        xaxis=dict(showgrid=True, gridcolor='#f0f0f0')
    )
    return fig


# ════════════════════════════════════════════════════════════════
# ── SIDEBAR — Input Form ──────────────────────────────────────────
# ════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🎓 Student Details")
    st.markdown("Fill in the student profile below:")
    st.markdown("---")

    st.markdown('<p class="section-header">📚 Academic Habits</p>',
                unsafe_allow_html=True)

    study_hours = st.slider(
        "Study Hours per Day", 0.5, 12.0, 6.0, 0.5,
        help="How many hours does the student study daily?"
    )
    attendance = st.slider(
        "Attendance Percentage (%)", 50.0, 100.0, 80.0, 1.0,
        help="Percentage of classes attended"
    )
    previous_score = st.slider(
        "Previous Exam Score", 0.0, 100.0, 65.0, 1.0,
        help="Score in the most recent exam"
    )
    tutoring = st.slider(
        "Tutoring Sessions per Week", 0, 5, 1,
        help="Number of extra tutoring sessions per week"
    )

    st.markdown("---")
    st.markdown('<p class="section-header">🌙 Lifestyle</p>',
                unsafe_allow_html=True)

    sleep_hours = st.slider(
        "Sleep Hours per Day", 3.0, 12.0, 7.0, 0.5,
        help="Average hours of sleep per night"
    )
    extracurricular = st.selectbox(
        "Extracurricular Activities",
        options=[("Yes", 1), ("No", 0)],
        format_func=lambda x: x[0]
    )
    internet = st.selectbox(
        "Internet Access at Home",
        options=[("Yes", 1), ("No", 0)],
        format_func=lambda x: x[0]
    )

    st.markdown("---")
    st.markdown('<p class="section-header">👨‍👩‍👦 Background</p>',
                unsafe_allow_html=True)

    parent_edu = st.selectbox(
        "Parent Education Level",
        ['No Formal Education', 'High School', 'Graduate', 'Post-Graduate']
    )
    motivation = st.selectbox(
        "Motivation Level",
        ['Low', 'Medium', 'High'],
        index=2
    )

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Score", use_container_width=True,
                             type="primary")


# ════════════════════════════════════════════════════════════════
# ── MAIN PAGE ─────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════

# Header
st.markdown("# 🎓 Student Performance Predictor")
st.markdown(
    "Enter student details in the **sidebar** and click **Predict Score** "
    "to get an AI-powered prediction of their exam performance."
)

if artifact is None:
    st.error("⚠️ Model not found. Please run `python -m src.train` first.")
    st.stop()

# ── Model info banner ─────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
metrics = artifact['metrics']

with col1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{artifact['model_name'].split()[0]}</div>
        <div class="metric-label">Model Type</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{metrics['R2']:.2f}</div>
        <div class="metric-label">R² Score (accuracy)</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">±{metrics['MAE']:.1f}</div>
        <div class="metric-label">Avg Error (MAE)</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{len(artifact['feature_names'])}</div>
        <div class="metric-label">Input Features</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# ── PREDICTION RESULT ─────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════

# Build input dict
input_data = {
    'study_hours_per_day'        : study_hours,
    'attendance_percentage'      : attendance,
    'previous_score'             : previous_score,
    'sleep_hours_per_day'        : sleep_hours,
    'extracurricular_activities' : extracurricular[1],
    'parent_education_level'     : parent_edu,
    'internet_access'            : internet[1],
    'tutoring_sessions_per_week' : tutoring,
    'motivation_level'           : motivation
}

# Always show live prediction (updates on slider change too)
result = predict(input_data)

if predict_btn:
    st.balloons()

# Result layout
left, right = st.columns([1, 1], gap="large")

with left:
    # Big result card
    st.markdown(f"""
    <div class="result-card">
        <div style="font-size:1rem; opacity:0.8; margin-bottom:0.5rem">
            PREDICTED EXAM SCORE
        </div>
        <div class="result-score">{result['score']}</div>
        <div class="result-grade">{result['grade']}</div>
        <div class="result-band">
            Confidence Band: {result['lower']} – {result['upper']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tip based on score
    if result['score'] < 55:
        st.markdown("""<div class="tip-box">
            💡 <b>Tip:</b> Increasing daily study hours and attendance
            has the highest impact on improving scores.
        </div>""", unsafe_allow_html=True)
    elif result['score'] < 75:
        st.markdown("""<div class="tip-box">
            💡 <b>Tip:</b> Adding tutoring sessions and improving
            motivation level can push the score into the A range.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="tip-box">
            ✅ <b>Great performance!</b> Maintaining consistency
            in study habits and attendance will sustain this level.
        </div>""", unsafe_allow_html=True)

with right:
    st.plotly_chart(
        make_gauge(result['score'], result['color']),
        use_container_width=True
    )

# ════════════════════════════════════════════════════════════════
# ── CHARTS ROW ───────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("### 📊 Student Profile Analysis")

chart_left, chart_right = st.columns(2, gap="large")

with chart_left:
    st.markdown("**Radar: Student vs Ideal Profile**")
    st.plotly_chart(make_radar(input_data), use_container_width=True)

with chart_right:
    st.markdown("**This Student vs Dataset Average**")
    st.plotly_chart(make_feature_bar(input_data), use_container_width=True)

# ════════════════════════════════════════════════════════════════
# ── EDA SECTION ──────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("### 🔬 Dataset Insights (EDA)")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Score Distribution",
    "🔥 Correlation Heatmap",
    "🎯 Feature Importance",
    "🤖 Model Comparison"
])

plot_map = {
    "tab1": "data/plot_target_distribution.png",
    "tab2": "data/plot_correlation_heatmap.png",
    "tab3": "data/plot_feature_importances_rf.png",
    "tab4": "data/plot_model_comparison.png",
}

with tab1:
    if os.path.exists(plot_map["tab1"]):
        st.image(plot_map["tab1"], use_column_width=True)
        st.markdown("""<div class="tip-box">
            📌 <b>Insight:</b> The final score is approximately normally
            distributed with mean ~62 — ideal for regression modeling.
        </div>""", unsafe_allow_html=True)

with tab2:
    if os.path.exists(plot_map["tab2"]):
        st.image(plot_map["tab2"], use_column_width=True)
        st.markdown("""<div class="tip-box">
            📌 <b>Insight:</b> study_hours, motivation, and previous_score
            show the strongest correlations with final_score.
        </div>""", unsafe_allow_html=True)

with tab3:
    if os.path.exists(plot_map["tab3"]):
        st.image(plot_map["tab3"], use_column_width=True)
        st.markdown("""<div class="tip-box">
            📌 <b>Insight:</b> Random Forest confirms study_hours and
            motivation_level as the top 2 most important features.
        </div>""", unsafe_allow_html=True)

with tab4:
    if os.path.exists(plot_map["tab4"]):
        st.image(plot_map["tab4"], use_column_width=True)
        st.markdown("""<div class="tip-box">
            📌 <b>Insight:</b> Linear Regression achieved the best R²=0.90
            — confirming that the feature-score relationships are linear.
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# ── FOOTER ───────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#94a3b8; font-size:0.85rem; padding:1rem'>
    Built by <b>Aryansingh Bais</b> · 
    Data Science with GenAI & Agentic AI · NareshIT Hyderabad ·
    <a href='https://github.com/Aryansingh-B' target='_blank'>GitHub</a> ·
    <a href='https://www.linkedin.com/in/aryansinghbais8/' target='_blank'>LinkedIn</a>
</div>
""", unsafe_allow_html=True)