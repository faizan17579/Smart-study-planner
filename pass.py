import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from catboost import CatBoostClassifier

# # Load the dataset
# data = pd.read_csv('data/student_exam_data.csv')

# # Print column names to confirm
# print("Columns in the dataset:", data.columns.tolist())

# # Check for missing values
# print("Missing values:\n", data.isnull().sum())

# # Handle missing values
# # For numerical columns, replace NaN with 0
# numerical_columns = ['Study Hours', 'Previous Exam Score']
# data[numerical_columns] = data[numerical_columns].fillna(0)

# # Define features and target
# X = data[['Study Hours', 'Previous Exam Score']]
# y = data['Pass/Fail']

# # Split data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train CatBoost Classifier
# catboost_model = CatBoostClassifier(
#     iterations=500,
#     depth=6,
#     learning_rate=0.1,
#     verbose=0,
#     random_state=42
# )

# catboost_model.fit(X_train, y_train)


# catboost_model.save_model("student_exam_model.cbm")
# print("Model saved as 'student_exam_model.cbm'.")


# # Make predictions
# train_preds = catboost_model.predict(X_train)
# test_preds = catboost_model.predict(X_test)

# # Evaluate with multiple classification metrics
# train_accuracy = accuracy_score(y_train, train_preds)
# test_accuracy = accuracy_score(y_test, test_preds)
# train_precision = precision_score(y_train, train_preds)
# test_precision = precision_score(y_test, test_preds)
# train_recall = recall_score(y_train, train_preds)
# test_recall = recall_score(y_test, test_preds)
# train_f1 = f1_score(y_train, train_preds)
# test_f1 = f1_score(y_test, test_preds)

# # Print results
# print("CatBoost Classification Performance:")
# print(f"Train Accuracy: {train_accuracy:.4f}")
# print(f"Test Accuracy: {test_accuracy:.4f}")
# print(f"Train Precision: {train_precision:.4f}")
# print(f"Test Precision: {test_precision:.4f}")
# print(f"Train Recall: {train_recall:.4f}")
# print(f"Test Recall: {test_recall:.4f}")
# print(f"Train F1-Score: {train_f1:.4f}")
# print(f"Test F1-Score: {test_f1:.4f}")

# # Cross-validation for more reliable estimate
# cv_scores = cross_val_score(catboost_model, X, y, cv=5, scoring='accuracy')
# print(f"Cross-Validated Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# # Feature importance
# feature_importance = catboost_model.get_feature_importance()
# for feature, importance in zip(X.columns, feature_importance):
#     print(f"{feature}: {importance:.4f}")

# test the model with new data
new_data = pd.DataFrame([{
    'Study Hours': 6,
    'Previous Exam Score': 80
}])


if __name__ == "__main__":
    # Load the model
    loaded_model = CatBoostClassifier()
    loaded_model.load_model("student_exam_model.cbm")

    # Make a prediction
    prediction = loaded_model.predict(new_data)[0]
    print(f"Prediction for new data: {prediction}")
    

