import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score, f1_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import joblib

# Load and prepare data
data = pd.read_csv('data/dataset.csv')

# Define features and target order
base_features = ['Hours_Studied', 'Attendance', 'Gender', 'Previous_Scores',
                'Tutoring_Sessions', 'Teacher_Quality', 'School_Type',
                'Parental_Education_Level']

target_sequence = ['Sleep_Hours', 'Physical_Activity', 'Extracurricular_Activities', 'Exam_Score']

# Initialize model storage
models = {}
X_all = data[base_features].copy()

# Preprocessing setup for XGBoost and RandomForest
categorical_features = ['Gender', 'Teacher_Quality', 'School_Type', 'Parental_Education_Level']
numerical_features = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Tutoring_Sessions']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

# Sequential modeling for first three targets
for i, target in enumerate(target_sequence[:-1]):  # Exclude Exam_Score
    print(f"\n=== Training model for {target} ===")

    # Prepare feature matrix
    if i == 0:
        X = X_all.copy()
    else:
        X = X_all.join(pd.DataFrame({
            f'pred_{prev_target}': models[prev_target]['predictions']
            for prev_target in target_sequence[:i]
        }))

    y = data[target].copy()

    # Handle classification case
    if target == 'Extracurricular_Activities':
        y = y.map({'Yes': 1, 'No': 0})
        y.fillna(0, inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)

        clf = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
        ])

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X)
        acc = accuracy_score(y_test, clf.predict(X_test))
        f1 = f1_score(y_test, clf.predict(X_test))

        print(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

        models[target] = {
            'model': clf,
            'predictions': y_pred,
            'accuracy': acc,
            'f1': f1
        }

        # Save the model
        joblib.dump(clf, f'model_{target}.pkl')

    else:
        y.fillna(y.median(), inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        reg = Pipeline([
            ('preprocessor', preprocessor),
            ('model', XGBRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                objective='reg:squarederror', random_state=42))
        ])

        reg.fit(X_train, y_train)
        y_pred = reg.predict(X)
        train_r2 = r2_score(y_train, reg.predict(X_train))
        test_r2 = r2_score(y_test, reg.predict(X_test))

        print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")

        models[target] = {
            'model': reg,
            'predictions': y_pred,
            'train_score': train_r2,
            'test_score': test_r2
        }

        # Save the model
        joblib.dump(reg, f'model_{target}.pkl')

# CatBoost for Exam_Score
print("\n=== Training CatBoost model for Exam_Score ===")

# Prepare feature matrix with predictions from previous models
X_exam = X_all.join(pd.DataFrame({
    f'pred_{target}': models[target]['predictions']
    for target in target_sequence[:-1]
}))

y_exam = data['Exam_Score'].copy()

# Handle missing values for CatBoost
numerical_columns = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Tutoring_Sessions', 
                    'pred_Sleep_Hours', 'pred_Physical_Activity']
categorical_columns = ['Gender', 'Teacher_Quality', 'School_Type', 'Parental_Education_Level', 
                      'pred_Extracurricular_Activities']

X_exam[numerical_columns] = X_exam[numerical_columns].fillna(0)
X_exam[categorical_columns] = X_exam[categorical_columns].fillna('Unknown')

# Convert pred_Extracurricular_Activities to categorical (string) for CatBoost
X_exam['pred_Extracurricular_Activities'] = X_exam['pred_Extracurricular_Activities'].map({1: 'Yes', 0: 'No'})

# Split data
X_train_exam, X_test_exam, y_train_exam, y_test_exam = train_test_split(
    X_exam, y_exam, test_size=0.2, random_state=42)

# Initialize and train CatBoost model
catboost_model = CatBoostRegressor(
    iterations=1000,
    depth=6,
    learning_rate=0.1,
    cat_features=categorical_columns,
    verbose=0,
    random_state=42
)
catboost_model.fit(X_train_exam, y_train_exam)

# Save the CatBoost model
catboost_model.save_model('catboost_exam_score.cbm')

# Make predictions
train_preds = catboost_model.predict(X_train_exam)
test_preds = catboost_model.predict(X_test_exam)

# Evaluate with multiple metrics
train_r2 = r2_score(y_train_exam, train_preds)
test_r2 = r2_score(y_test_exam, test_preds)
mse_train = mean_squared_error(y_train_exam, train_preds)
mse_test = mean_squared_error(y_test_exam, test_preds)
mae_train = mean_absolute_error(y_train_exam, train_preds)
mae_test = mean_absolute_error(y_test_exam, test_preds)

# Print results
print("CatBoost Performance for Exam_Score:")
print(f"Train R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Train MSE: {mse_train:.4f}")
print(f"Test MSE: {mse_test:.4f}")
print(f"Train MAE: {mae_train:.4f}")
print(f"Test MAE: {mae_test:.4f}")

# Cross-validation
cv_scores = cross_val_score(catboost_model, X_exam, y_exam, cv=5, scoring='r2')
print(f"Cross-Validated R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Feature importance
feature_importance = catboost_model.get_feature_importance()
for feature, importance in zip(X_exam.columns, feature_importance):
    print(f"{feature}: {importance:.4f}")

# Final performance summary
print("\n=== Final Model Performance ===")
for target in target_sequence[:-1]:
    if target == 'Extracurricular_Activities':
        print(f"{target} - Accuracy: {models[target]['accuracy']:.4f}, F1 Score: {models[target]['f1']:.4f}")
    else:
        print(f"{target} - Train R²: {models[target]['train_score']:.4f}, Test R²: {models[target]['test_score']:.4f}")
print(f"Exam_Score - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")

print("\nAll models saved successfully.")