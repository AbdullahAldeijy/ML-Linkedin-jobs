#  Fast LinkedIn Jobs ML Pipeline

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

print("Starting Fast LinkedIn Jobs ML Pipeline...")

# Load data
df = pd.read_csv("data_jobs.csv")
print(f"Data loaded: {df.shape}")

# Handle missing values in target
df = df.dropna(subset=['level'])
print(f"After removing missing targets: {df.shape}")

# Feature Engineering
print("\n=== Feature Engineering ===")

# Encode Regions
le_regions = LabelEncoder()
df['Regions'] = le_regions.fit_transform(df['Regions'])
print(f"Encoded {len(le_regions.classes_)} regions")

# Encode target variable
level_map = {
    'Not Applicable': 0, 'Internship': 1, 'Entry level': 2,
    'Associate': 3, 'Mid-Senior level': 4, 'Director': 5, 'Executive': 6
}
df['level'] = df['level'].map(level_map)

# One-hot encode industries (limit to top 20 to speed up)
top_industries = df['industries'].value_counts().head(20).index
df['industries_simplified'] = df['industries'].apply(lambda x: x if x in top_industries else 'Other')

one_hot = OneHotEncoder(handle_unknown='ignore')
industries_encoded = one_hot.fit_transform(df[['industries_simplified']]).toarray()
industry_cols = [f'industry_{cat}' for cat in one_hot.categories_[0]]

# Add encoded industries to dataframe
for i, col in enumerate(industry_cols):
    df[col] = industries_encoded[:, i]

# Prepare features for modeling
feature_cols = ['Regions', 'year_of_ex', 'description_length', 'is_remote'] + industry_cols
X = df[feature_cols]
y = df['level']

print(f"Features shape: {X.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain set: {X_train_scaled.shape}")
print(f"Test set: {X_test_scaled.shape}")

# Train baseline models
print("\n=== Training Models ===")

# Logistic Regression
print("Training Logistic Regression...")
log_clf = LogisticRegression(random_state=42, max_iter=1000)
log_clf.fit(X_train_scaled, y_train)
y_pred_log = log_clf.predict(X_test_scaled)
log_acc = accuracy_score(y_test, y_pred_log)
print(f"Logistic Regression Accuracy: {log_acc:.4f}")

# Random Forest
print("Training Random Forest...")
rf_clf = RandomForestClassifier(n_estimators=50, random_state=42)  # Reduced trees for speed
rf_clf.fit(X_train_scaled, y_train)
y_pred_rf = rf_clf.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {rf_acc:.4f}")

# Apply SMOTE
print("\nApplying SMOTE...")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
print(f"After SMOTE: {X_train_res.shape}")

# Train with SMOTE
print("Training Random Forest with SMOTE...")
rf_smote = RandomForestClassifier(n_estimators=50, random_state=42)
rf_smote.fit(X_train_res, y_train_res)
y_pred_rf_smote = rf_smote.predict(X_test_scaled)
rf_smote_acc = accuracy_score(y_test, y_pred_rf_smote)
print(f"Random Forest + SMOTE Accuracy: {rf_smote_acc:.4f}")

# Save models
print("\n=== Saving Models ===")
joblib.dump(scaler, "scaler_jobs_level.pkl")
joblib.dump(rf_smote, "Random_Forest_level_model.pkl")

# Save level mapping
level_mapping = {v: k for k, v in level_map.items()}
with open("level_mapping.json", "w") as f:
    json.dump(level_mapping, f)

# Save feature names
feature_info = {
    'feature_names': feature_cols,
    'region_classes': le_regions.classes_.tolist()
}
with open("feature_info.json", "w") as f:
    json.dump(feature_info, f)

print("Saved: scaler_jobs_level.pkl, Random_Forest_level_model.pkl")
print("Saved: level_mapping.json, feature_info.json")

# Performance Summary
print("\n" + "="*50)
print("MODEL PERFORMANCE SUMMARY")
print("="*50)
print(f"Logistic Regression      : {log_acc:.4f}")
print(f"Random Forest           : {rf_acc:.4f}")
print(f"Random Forest + SMOTE   : {rf_smote_acc:.4f}")

print(f"\nBest Model: Random Forest + SMOTE")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_rf_smote))

print("\nScript completed successfully!")