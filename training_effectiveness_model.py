# file: training_effectiveness_model.py
# Purpose: Predict training effectiveness using Random Forest and XGBoost on synthetic data

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Generate Synthetic Data
np.random.seed(42)
n_samples = 500

data = pd.DataFrame({
    "trainee_experience_years": np.random.randint(0, 15, n_samples),
    "training_duration_hours": np.random.randint(1, 8, n_samples),
    "engagement_score": np.random.uniform(0.3, 1.0, n_samples),
    "pre_training_score": np.random.uniform(30, 70, n_samples),
    "course_difficulty": np.random.choice(['Easy', 'Medium', 'Hard'], n_samples),
    "instructor_rating": np.random.uniform(2.5, 5.0, n_samples)
})

# Create target: post-training score improvement
difficulty_map = {'Easy': 1.2, 'Medium': 1.0, 'Hard': 0.8}
data['difficulty_factor'] = data['course_difficulty'].map(difficulty_map)

# Simulate target variable: training_effectiveness (score improvement)
data['training_effectiveness'] = (
    (data['engagement_score'] * 20 +
     data['training_duration_hours'] * 1.5 +
     data['instructor_rating'] * 2) * data['difficulty_factor']
)

# 2. Preprocessing
data = pd.get_dummies(data.drop(columns=['difficulty_factor']), drop_first=True)

# 3. Split the data
X = data.drop(columns='training_effectiveness')
y = data['training_effectiveness']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# 5. Evaluate Models
def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{name} RMSE: {rmse:.2f}, R2 Score: {r2:.2f}")
    return y_pred

rf_preds = evaluate_model(rf_model, X_test, y_test, "Random Forest")
xgb_preds = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

# 6. Feature Importance
def plot_feature_importance(model, features, title):
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance[sorted_idx], y=np.array(features)[sorted_idx])
    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_feature_importance(rf_model, X.columns, "Random Forest Feature Importance")
plot_feature_importance(xgb_model, X.columns, "XGBoost Feature Importance")
