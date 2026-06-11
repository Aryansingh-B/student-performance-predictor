
# Generate a realistic synthetic student performance dataset
# with meaningful correlations between features and final score.

import numpy as np
import pandas as pd

#  Reproducibility
np.random.seed(42)
N = 1000  # number of students

# Independent Features 
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

# Target Variable: final_score
# Built with realistic weighted contributions from each feature
# so EDA reveals genuine correlations (not random noise)

# Encode categorical features for score formula
parent_edu_score = {
    'No Formal Education': 0,
    'High School': 1,
    'Graduate': 2,
    'Post-Graduate': 3
}
motivation_score = {'Low': 0, 'Medium': 1, 'High': 2}

parent_enc   = np.array([parent_edu_score[p] for p in parent_education])
motivation_enc = np.array([motivation_score[m] for m in motivation_level])

# Weighted formula — each feature contributes meaningfully
base_score = (
    study_hours        * 3.5   +   # strongest predictor
    attendance         * 0.25  +   # consistent attendance matters
    previous_score     * 0.30  +   # past performance carries weight
    sleep_hours        * 1.2   +   # healthy sleep boosts performance
    extracurricular    * 2.0   +   # slight positive effect
    internet_access    * 2.5   +   # resource access helps
    tutoring_sessions  * 1.8   +   # tutoring has clear benefit
    parent_enc         * 1.5   +   # parental education influence
    motivation_enc     * 3.0       # motivation is highly impactful
)

# Normalize to 0–100 range
min_s, max_s = base_score.min(), base_score.max()
normalized = (base_score - min_s) / (max_s - min_s) * 70 + 25  # range 25–95

# Add realistic noise
noise = np.random.normal(0, 3.5, N)
final_score = np.clip(np.round(normalized + noise, 1), 0, 100)

# Assemble DataFrame 
df = pd.DataFrame({
    'study_hours_per_day'      : study_hours,
    'attendance_percentage'    : attendance,
    'previous_score'           : previous_score,
    'sleep_hours_per_day'      : sleep_hours,
    'extracurricular_activities': extracurricular,
    'parent_education_level'   : parent_education,
    'internet_access'          : internet_access,
    'tutoring_sessions_per_week': tutoring_sessions,
    'motivation_level'         : motivation_level,
    'final_score'              : final_score
})

# Save
df.to_csv('data/student_data.csv', index=False)

print(f" Dataset generated: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n Final Score Stats:")
print(df['final_score'].describe().round(2))
print(f"\n Sample rows:")
print(df.head(3).to_string())