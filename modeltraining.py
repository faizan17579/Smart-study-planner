import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score, f1_score, mean_squared_error, mean_absolute_error, confusion_matrix # Added confusion_matrix
from catboost import CatBoostRegressor
import joblib

# Load and prepare data
data = pd.read_csv('data/dataset.csv')

# Define features and target order
base_features = ['Hours_Studied', 'Attendance', 'Gender', 'Previous_Scores',
                'Tutoring_Sessions', 'Teacher_Quality', 'School_Type',
                'Parental_Education_Level', 'Sleep_Hours', 'Physical_Activity']

target_sequence = ['Extracurricular_Activities', 'Exam_Score']

# Initialize model storage
models = {}
X_all = data[base_features].copy()

# Preprocessing setup
categorical_features = ['Gender', 'Teacher_Quality', 'School_Type', 'Parental_Education_Level']
numerical_features = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Tutoring_Sessions',
                     'Sleep_Hours', 'Physical_Activity']
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

# Fit scaler for Sleep_Hours and Physical_Activity and save it
scaler = StandardScaler()
scaler.fit(data[['Sleep_Hours', 'Physical_Activity']])
joblib.dump(scaler, 'scaler.pkl')

# Sequential modeling for targets
for i, target in enumerate(target_sequence):
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

        # Compute and display confusion matrix
        y_pred_test = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_test)
        print(f"Confusion Matrix for {target}:")
        print(cm)
        print(f"True Positives (Yes predicted correctly): {cm[1, 1]}")
        print(f"True Negatives (No predicted correctly): {cm[0, 0]}")
        print(f"False Positives (No predicted as Yes): {cm[0, 1]}")
        print(f"False Negatives (Yes predicted as No): {cm[1, 0]}")

        print(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

        models[target] = {
            'model': clf,
            'predictions': y_pred,
            'accuracy': acc,
            'f1': f1
        }

        joblib.dump(clf, f'model_{target}.pkl')

    else:  # Exam_Score
        y.fillna(y.median(), inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        numerical_columns = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Tutoring_Sessions',
                            'Sleep_Hours', 'Physical_Activity']
        categorical_columns = ['Gender', 'Teacher_Quality', 'School_Type', 'Parental_Education_Level',
                              'pred_Extracurricular_Activities']

        X_train_cat = X_train.copy()
        X_test_cat = X_test.copy()
        X_cat = X.copy()

        X_cat[numerical_columns] = X_cat[numerical_columns].fillna(0)
        X_cat[categorical_columns] = X_cat[categorical_columns].fillna('Unknown')
        X_train_cat[numerical_columns] = X_train_cat[numerical_columns].fillna(0)
        X_train_cat[categorical_columns] = X_train_cat[categorical_columns].fillna('Unknown')
        X_test_cat[numerical_columns] = X_test_cat[numerical_columns].fillna(0)
        X_test_cat[categorical_columns] = X_test_cat[categorical_columns].fillna('Unknown')

        if i > 0:
            X_cat['pred_Extracurricular_Activities'] = X_cat['pred_Extracurricular_Activities'].map({1: 'Yes', 0: 'No'})
            X_train_cat['pred_Extracurricular_Activities'] = X_train_cat['pred_Extracurricular_Activities'].map({1: 'Yes', 0: 'No'})
            X_test_cat['pred_Extracurricular_Activities'] = X_test_cat['pred_Extracurricular_Activities'].map({1: 'Yes', 0: 'No'})

        catboost_model = CatBoostRegressor(
            iterations=1000,
            depth=6,
            learning_rate=0.1,
            cat_features=categorical_columns,
            verbose=0,
            early_stopping_rounds=50,
            random_state=42
        )
        catboost_model.fit(X_train_cat, y_train)

        catboost_model.save_model('catboost_exam_score.cbm')

        train_preds = catboost_model.predict(X_train_cat)
        test_preds = catboost_model.predict(X_test_cat)
        y_pred = catboost_model.predict(X_cat)

        train_r2 = r2_score(y_train, train_preds)
        test_r2 = r2_score(y_test, test_preds)
        mse_train = mean_squared_error(y_train, train_preds)
        mse_test = mean_squared_error(y_test, test_preds)
        mae_train = mean_absolute_error(y_train, train_preds)
        mae_test = mean_absolute_error(y_test, test_preds)

        print("CatBoost Performance for Exam_Score:")
        print(f"Train R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Train MSE: {mse_train:.4f}")
        print(f"Test MSE: {mse_test:.4f}")
        print(f"Train MAE: {mae_train:.4f}")
        print(f"Test MAE: {mae_test:.4f}")

        models[target] = {
            'model': catboost_model,
            'predictions': y_pred,
            'train_score': train_r2,
            'test_score': test_r2
        }

        cv_scores = cross_val_score(catboost_model, X_cat, y, cv=10, scoring='r2')
        print(f"Cross-Validated R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        feature_importance = catboost_model.get_feature_importance()
        for feature, importance in zip(X_cat.columns, feature_importance):
            print(f"{feature}: {importance:.4f}")

# Final performance summary
print("\n=== Final Model Performance ===")
for target in target_sequence:
    if target == 'Extracurricular_Activities':
        print(f"{target} - Accuracy: {models[target]['accuracy']:.4f}, F1 Score: {models[target]['f1']:.4f}")
    else:
        print(f"{target} - Train R²: {models[target]['train_score']:.4f}, Test R²: {models[target]['test_score']:.4f}")

print("\nAll models saved successfully.")