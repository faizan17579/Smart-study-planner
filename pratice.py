import joblib
import pandas as pd

# Example input (replace with real values)
new_data = pd.DataFrame([{
    'Hours_Studied': 60,
    'Attendance': 90,
    'Gender': 'Male',
    'Previous_Scores': 95,
    'Tutoring_Sessions': 2,
    'Teacher_Quality': 'High',
    'School_Type': 'Public',
    'Parental_Education_Level': 'Bachelor'
}])

# Optional feature engineering if used in training
new_data['Study_Attend_Interaction'] = new_data['Hours_Studied'] * new_data['Attendance']

# Load and predict sequentially
targets = ['Sleep_Hours', 'Physical_Activity', 'Extracurricular_Activities', 'Exam_Score']
for target in targets:
    model = joblib.load(f'model_{target}.pkl')
    prediction = model.predict(new_data)[0]
    print(f'{target} prediction: {prediction}')
    
    # Add prediction as input for next model
    new_data[f'pred_{target}'] = prediction
