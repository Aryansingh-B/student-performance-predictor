# Generate a realistic synthetic student performance dataset
# with meaningful correlations between features and final score.

import numpy as np
import pandas as pd

#  Reproducibility
np.random.seed(42)
N = 1000  # number of students

# ── Independent Features ──────────────────────────────────────────
study_hours        = np.round(np.random.uniform(1, 10, N), 1)
attendance         = np.round(np.random.uniform(50, 100, N), 1)
previous_score     = np.round(np.random.uniform(30, 95, N), 1)
sleep_hours        = np.round(np.random.uniform(4, 10, N), 1)
extracurricular    = np.random.randint(0, 2, N)          # 0 = No, 1 = Yes
internet_access    = np.random.randint(0, 2, N)          # 0 = No, 1 = Yes
tutoring_sessions  = np.random.randint(0, 6, N)          # 0–5 per week

parent_education   = np.random.choice(
    ['No Formal Education', 'High School', 'Graduate', 'Post-Graduate'],
    N, p=[0.10, 0.35, 0.40, 0.15]
)

motivation_level   = np.random.choice(
    ['Low', 'Medium', 'High'],
    N, p=[0.25, 0.45, 0.30]
)

# ── Encode Categorical Features ───────────────────────────────────
parent_edu_map = {
    'No Formal Education': 0,
    'High School':         1,
    'Graduate':            2,
    'Post-Graduate':       3
}

# KEY FIX: motivation is now a MULTIPLIER not an additive value
# Low motivation = 0.55x, Medium = 0.85x, High = 1.15x
# This means a lazy student with high study hours still gets penalized
motivation_multiplier_map = {
    'Low':    0.55,
    'Medium': 0.85,
    'High':   1.15
}

parent_enc          = np.array([parent_edu_map[p]              for p in parent_education])
motivation_mult     = np.array([motivation_multiplier_map[m]   for m in motivation_level])

# ── Score Formula ─────────────────────────────────────────────────
# study_hours and previous_score are multiplied by motivation
# so Low motivation genuinely hurts — not just a small additive penalty

base_score = (
    study_hours       * 5.5  * motivation_mult  +   # motivation scales effort directly
    attendance        * 0.30                    +   # attendance: 50%→15pts, 100%→30pts
    previous_score    * 0.25 * motivation_mult  +   # past performance scaled by drive
    sleep_hours       * 1.2                     +   # healthy sleep: 4h→5pts, 9h→11pts
    extracurricular   * 2.5                     +   # slight positive effect
    internet_access   * 3.0                     +   # resource access
    tutoring_sessions * 2.5                     +   # 0 sessions→0, 5 sessions→12.5pts
    parent_enc        * 2.0                         # 0→0, 3→6pts
)

# Add realistic noise BEFORE clipping
noise = np.random.normal(0, 4.0, N)
final_score = np.clip(np.round(base_score + noise, 1), 0, 100)

# ── Assemble DataFrame ────────────────────────────────────────────
df = pd.DataFrame({
    'study_hours_per_day'       : study_hours,
    'attendance_percentage'     : attendance,
    'previous_score'            : previous_score,
    'sleep_hours_per_day'       : sleep_hours,
    'extracurricular_activities': extracurricular,
    'parent_education_level'    : parent_education,
    'internet_access'           : internet_access,
    'tutoring_sessions_per_week': tutoring_sessions,
    'motivation_level'          : motivation_level,
    'final_score'               : final_score
})

# ── Save ──────────────────────────────────────────────────────────
df.to_csv('data/student_data.csv', index=False)

print(f"✅ Dataset generated: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n📊 Final Score Stats:")
print(df['final_score'].describe().round(2))

# Show score breakdown by motivation — this proves the fix works
print(f"\n🎯 Average Score by Motivation Level:")
print(df.groupby('motivation_level')['final_score'].mean().round(1).to_string())

print(f"\n📋 Sample rows:")
print(df.head(3).to_string())