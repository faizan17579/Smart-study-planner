import matplotlib
matplotlib.use('Agg')
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostRegressor
from catboost import CatBoostClassifier
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import matplotlib.pyplot as plt

# Load the trained models and scaler
models = {
    'Extracurricular_Activities': joblib.load('model_Extracurricular_Activities.pkl'),
    'Exam_Score': CatBoostRegressor().load_model('catboost_exam_score.cbm')
}
scaler = joblib.load('scaler.pkl')

exam=CatBoostClassifier().load_model('student_exam_model.cbm')

def train_classifier(file_path, username):
    data = pd.read_csv(file_path)
    X = data[[col for col in data.columns if col.endswith('_Score')]]
    y = data['Weak_Subject'] if 'Weak_Subject' in data.columns else pd.Series(['Math'] * len(data))
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)
    return clf

def train_score_predictor(file_path, username):
    return models

def predict_weak_subject(clf, scores, username):
    scores_df = pd.DataFrame([scores])
    prediction = clf.predict(scores_df)[0]
    return prediction.lower()  # Normalize to lowercase

def predict_future_score_and_health(score_predictor, features, username):
    initial_features = {
        'Hours_Studied': features['Study_Time'],
        'Attendance': features['Attendance'],
        'Gender': features.get('Gender', 'Male'),
        'Previous_Scores': features['Previous_Exam_Score'],
        'Tutoring_Sessions': features['Tutoring_Sessions'],
        'Teacher_Quality': features.get('Teacher_Quality', 'Medium'),
        'School_Type': features.get('School_Type', 'Public'),
        'Parental_Education_Level': features.get('Parental_Education_Level', 'High School'),
        'Sleep_Hours': features.get('Sleep_Hours', 7.0),
        'Physical_Activity': features.get('Physical_Activity', 3.0)
    }
    X = pd.DataFrame([initial_features])

    X[['Sleep_Hours', 'Physical_Activity']] = scaler.transform(X[['Sleep_Hours', 'Physical_Activity']])

    predictions = {}
    target_sequence = ['Extracurricular_Activities', 'Exam_Score']
    
    for target in target_sequence:
        if target == 'Extracurricular_Activities':
            pred = score_predictor[target].predict(X)[0]
            predictions[target] = pred
        else:
            X['pred_Extracurricular_Activities'] = X['pred_Extracurricular_Activities'].map({1: 'Yes', 0: 'No'})
            numerical_columns = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Tutoring_Sessions',
                                'Sleep_Hours', 'Physical_Activity']
            categorical_columns = ['Gender', 'Teacher_Quality', 'School_Type', 'Parental_Education_Level',
                                 'pred_Extracurricular_Activities']
            X[numerical_columns] = X[numerical_columns].fillna(0)
            X[categorical_columns] = X[categorical_columns].fillna('Unknown')
            pred = float(score_predictor[target].predict(X)[0])
            predictions[target] = pred
        
        if target in target_sequence[:-1]:
            X[f'pred_{target}'] = pred

    predictions['Extracurricular_Activities'] = 'Yes' if predictions['Extracurricular_Activities'] == 1 else 'No'

    return {
        'Future_Score': predictions['Exam_Score'],
        'Sleep_Hours': initial_features['Sleep_Hours'],
        'Extracurricular_Activities': predictions['Extracurricular_Activities'],
        'Physical_Activity': initial_features['Physical_Activity']
    }

def recommend_study_hours(clf, scores, username, occupied_hours):
    scores_df = pd.DataFrame([scores])
    weak_subject = clf.predict(scores_df)[0].lower()  # Normalize to lowercase
    print(f"Occupied hours: {occupied_hours}")

    score_file = os.path.join('scores', f'f.{username}.csv')
    score_history = []
    if os.path.exists(score_file):
        df = pd.read_csv(score_file)
        score_history = df.drop(columns=['timestamp']).to_dict('records')

    print(f"Score history for {username}: {score_history}")

    subject_map = {f"{col.capitalize()}_Score": col.lower() for col in ['math', 'physics', 'chemistry', 'english', 'history']}
    reverse_map = {v: k for k, v in subject_map.items()}
    subjects = list(set(subject_map.get(subj, subj.lower()) for subj in scores.keys()))  # Ensure unique subjects
    print(f"Adjusted subjects for querying: {subjects}")

    predicted_scores = []
    trends = []
    plot_files = []

    plot_dir = os.path.join('plots', username)
    os.makedirs(plot_dir, exist_ok=True)

    if score_history and len(score_history) > 1:
        for subj in subjects:
            historical_scores = [entry.get(subj, 0) for entry in score_history]
            print(f"Subject: {subj}, Historical scores: {historical_scores}")

            if len(set(historical_scores)) == 1:
                print(f"All scores same for {subj}, skipping regression.")
                pred_score = historical_scores[-1]
                trend = 0.0
            else:
                X = np.array(range(len(historical_scores))).reshape(-1, 1)
                y = np.array(historical_scores)

                lr = LinearRegression()
                lr.fit(X, y)

                next_step = np.array([[len(historical_scores)]])
                pred_score = lr.predict(next_step)[0]
                trend = lr.coef_[0]

            predicted_scores.append(max(0, min(100, pred_score)))
            trends.append(trend)

            plt.figure(figsize=(8, 6))
            submission_numbers = list(range(1, len(historical_scores) + 1))
            plt.plot(submission_numbers, historical_scores, marker='o', label=f"{subj.capitalize()} Scores")

            plt.plot(
                submission_numbers + [submission_numbers[-1] + 1],
                historical_scores + [pred_score],
                linestyle='--',
                color='red',
                label='Predicted Trend'
            )

            plt.ylim(0, 100)
            plt.yticks(np.arange(0, 101, 20))
            plt.xticks(submission_numbers + [submission_numbers[-1] + 1])

            plt.axhspan(90, 100, facecolor='green', alpha=0.1, label='Good (90-100)')
            plt.axhspan(65, 89, facecolor='lightblue', alpha=0.1, label='Stable (65-89)')
            plt.axhspan(40, 64, facecolor='khaki', alpha=0.1, label='Average (40-64)')
            plt.axhspan(0, 25, facecolor='red', alpha=0.1, label='Poor (0-25)')

            latest_score = historical_scores[-1]
            if latest_score >= 90:
                status = "Good"
            elif latest_score >= 65:
                status = "Stable"
            elif latest_score >= 40:
                status = "Average"
            elif latest_score < 25:
                status = "Poor"
            else:
                status = "Needs Improvement"

            plt.text(
                submission_numbers[-1],
                latest_score,
                f' ({status})',
                fontsize=10,
                ha='left',
                va='bottom',
                color='black'
            )

            trend_status = "Declining" if trend < -0.1 else "Improving" if trend > 0.1 else "Stable"
            plt.title(f"Score Trend for {subj.capitalize()} ({trend_status})")
            plt.xlabel("Submission Number")
            plt.ylabel("Score")
            plt.legend(loc='upper left', fontsize=9)
            plt.grid(True, linestyle='--', alpha=0.5)

            plot_filename = os.path.join(plot_dir, f"{subj.capitalize()}.png")
            plt.savefig(plot_filename, bbox_inches='tight')
            plt.close()
            plot_basename = f"{subj.capitalize()}.png"
            if os.path.exists(plot_filename):
                print(f"Successfully saved plot: {plot_filename}")
                plot_files.append(plot_basename)
            else:
                print(f"Failed to save plot: {plot_filename}")
    else:
        predicted_scores = [scores[subj] for subj in scores.keys()]
        trends = [0] * len(subjects)

    total_slots = occupied_hours.get('total_slots', 168)
    total_occupied_hours = (occupied_hours.get('total_school_slots', 0) +
                           occupied_hours.get('total_sleep_hours', 0) +
                           occupied_hours.get('total_physical_activity_hours', 0) +
                           occupied_hours.get('total_study_hours', 0))
    available_hours = max(0, total_slots - total_occupied_hours)
    print(f"Total slots: {total_slots}")
    print(f"Total occupied hours: {total_occupied_hours}")
    print(f"Available hours: {available_hours}")

    recommendations = {}
    target_score = 100
    base_hours = 1

    for i, subj in enumerate(subjects):
        original_subj = reverse_map.get(subj, f"{subj}_Score")
        current_score = scores.get(original_subj, 0)
        pred_score = predicted_scores[i]
        trend = trends[i]

        hours = base_hours
        if trend < 0:
            hours += 3
        if current_score < target_score:
            hours += max(0, (target_score - current_score) // 10)
        recommendations[subj] = max(2, min(10, hours))

    recommendations[weak_subject] = max(recommendations.get(weak_subject, 5), 10)

    total_recommended_hours = sum(recommendations.values())
    print(f"Initial total recommended hours: {total_recommended_hours}")

    if total_recommended_hours > available_hours:
        weak_subject_hours = recommendations[weak_subject]
        other_hours = total_recommended_hours - weak_subject_hours
        if other_hours > 0 and (available_hours - weak_subject_hours) > 0:
            scale_factor = (available_hours - weak_subject_hours) / other_hours
            for subj in recommendations:
                if subj != weak_subject:
                    recommendations[subj] = max(2, int(recommendations[subj] * scale_factor))
        elif available_hours < weak_subject_hours:
            recommendations[weak_subject] = max(2, int(available_hours))
            for subj in recommendations:
                if subj != weak_subject:
                    recommendations[subj] = 2  # Minimum hours for others

    total_recommended_hours = sum(recommendations.values())
    if total_recommended_hours > available_hours:
        scale_factor = available_hours / total_recommended_hours
        for subj in recommendations:
            recommendations[subj] = max(2, int(recommendations[subj] * scale_factor))

    total_recommended_hours = sum(recommendations.values())
    total_all_hours = occupied_hours.get('total_study_hours', 0) + total_recommended_hours
    if total_all_hours > total_slots:
        scale_factor = (total_slots - occupied_hours.get('total_study_hours', 0)) / total_recommended_hours
        for subj in recommendations:
            recommendations[subj] = max(2, int(recommendations[subj] * scale_factor))

    print(f"Final total recommended hours: {sum(recommendations.values())}")
    print(f"Total all hours (existing + recommended): {total_all_hours}")
    print(f"Recommendations: {recommendations}")
    return recommendations, plot_files

def predpassorfail(studyperday, previousscores):
    #  round off the input values to the nearest integer
    studyperday = round(studyperday)
    previousscores = round(previousscores)
    print(f"Study hours per day: {studyperday}, Previous exam score: {previousscores}")
    # Prepare the input features as a DataFrame
    features = pd.DataFrame({
        'Study Hours': [studyperday],
        'Previous Exam Score': [previousscores]
    })
    
    
    # Predict using the loaded CatBoost model
    prediction = exam.predict(features)[0]
    
    print(f"Prediction for new data: {prediction}")
    
    if prediction == 1 or prediction == '1':
        result = "Pass"
    else:
        result = "Fail"
    
    
    return result

    
