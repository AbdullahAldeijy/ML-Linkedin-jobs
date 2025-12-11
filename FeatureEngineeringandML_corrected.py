#  استيراد المكتبات اللازمة

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    classification_report, confusion_matrix
)

# Feature Engineering
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# نماذج التعلم الآلي
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# معالجة عدم توازن الفئات
from imblearn.over_sampling import SMOTE

# للتصوير
from matplotlib import pyplot as plt

# لحفظ النموذج
import joblib
import json

import warnings
warnings.filterwarnings("ignore")

print("Starting LinkedIn Jobs ML Pipeline...")

# === CODE BLOCK SEPARATOR ===

#قراءة البيانات واستكشافها
df = pd.read_csv("data_jobs.csv")
print("Data loaded successfully!")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# === CODE BLOCK SEPARATOR ===

# حذف عمود Unnamed: 0 لأنه مجرد Index قديم
if 'Unnamed: 0' in df.columns:
    df = df.drop(['Unnamed: 0'], axis=1)

print("\nData info:")
df.info()

# === CODE BLOCK SEPARATOR ===

# القيم المفقودة
print("\nMissing values:")
print(df.isna().sum())

# === CODE BLOCK SEPARATOR ===

# Handle missing values in level column (target variable)
print(f"\nBefore handling missing values in 'level': {df['level'].isna().sum()}")
df = df.dropna(subset=['level'])  # Remove rows with missing target values
print(f"After handling missing values in 'level': {df['level'].isna().sum()}")
print(f"New shape: {df.shape}")

# === CODE BLOCK SEPARATOR ===

#  مصفوفة الارتباط للمتغيرات العددية
print("\n=== Correlation Analysis ===")
# اختيار الأعمدة العددية فقط
num_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(12, 8))
corr = num_df.corr()
sns.heatmap(corr, annot=True, cmap="Blues", linewidths=0.3)
plt.title("Correlation Matrix of Numerical Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# === CODE BLOCK SEPARATOR ===

# جدول عددي يوضح عدد كل فئة بالنسبة لمستوى الوظيفة
print("\n=== Target Variable Analysis ===")
level_counts = df["level"].value_counts().rename_axis("level").reset_index(name="count")
level_counts["percent"] = (level_counts["count"] / len(df) * 100).round(2)

print("Distribution of Job Levels:")
print(level_counts)

# رسم عمودي لتوزيع مستويات الوظيفة
plt.figure(figsize=(10,6))
sns.countplot(data=df, x="level", order=df["level"].value_counts().index)
plt.title("Distribution of Job Levels (Target Variable)", fontsize=14, fontweight="bold")
plt.xlabel("Job Level")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === CODE BLOCK SEPARATOR ===

# توزيع الوظائف حسب المنطقة
print("\n=== Regional Analysis ===")
reg_counts = df["Regions"].value_counts().rename_axis("Region").reset_index(name="count")
reg_counts["percent"] = (reg_counts["count"] / len(df) * 100).round(2)

print("Distribution of Jobs by Region:")
print(reg_counts)

plt.figure(figsize=(10,6))
sns.countplot(data=df, x="Regions", order=df["Regions"].value_counts().index)
plt.title("Distribution of Jobs by Region", fontsize=14, fontweight="bold")
plt.xlabel("Region")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === CODE BLOCK SEPARATOR ===

#مستوى الوظيفة داخل كل منطقة
plt.figure(figsize=(14,8))
sns.countplot(data=df, x="Regions", hue="level", order=df["Regions"].value_counts().index)
plt.title("Job Levels Distribution per Region", fontsize=14, fontweight="bold")
plt.xlabel("Region")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.legend(title="Job Level", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# === CODE BLOCK SEPARATOR ===

#  Feature Engineering
print("\n=== Feature Engineering ===")

# ترميز Regions باستخدام LabelEncoder
print("Regions value counts:")
print(df['Regions'].value_counts())

le_regions = LabelEncoder()
df['Regions'] = le_regions.fit_transform(df['Regions'])
print("Region classes encoded successfully - total regions:", len(le_regions.classes_))

# === CODE BLOCK SEPARATOR ===

# عرض القيم الفئوية في level
print("\nLevel value counts:")
print(df['level'].value_counts())

# ترميز level (هذه هي الـ Target التي سندرب عليها)
level_map = {
    'Not Applicable': 0,
    'Internship': 1,
    'Entry level': 2,
    'Associate': 3,
    'Mid-Senior level': 4,
    'Director': 5,
    'Executive': 6
}
df['level'] = df['level'].map(level_map)
print("Encoded levels:")
print(df[['level']].head())

# === CODE BLOCK SEPARATOR ===

# استخدام OneHotEncoding على عمود industries
print("\n=== One-Hot Encoding Industries ===")
col_name = ['industries']

one_hot = OneHotEncoder(handle_unknown='ignore')
one_hot_array = one_hot.fit_transform(df[col_name]).toarray()

# أسماء الأعمدة الجديدة
column_names = []
for i in range(len(one_hot.categories_)):
    for j in range(len(one_hot.categories_[i])):
        column_names.append(col_name[i] + '_' + one_hot.categories_[i][j])

print(f"OneHot columns: {len(column_names)}, first 10: {column_names[:10]}")

# === CODE BLOCK SEPARATOR ===

# تحويل الناتج إلى DataFrame
oh_df = pd.DataFrame(one_hot_array, index=df.index, columns=column_names)

# إلحاق الأعمدة الجديدة بـ df
for col in oh_df.columns:
    df[col] = oh_df[col]

# حذف العمود الأصلي industries
df.drop(col_name, axis=1, inplace=True)

print("Data after OneHot encoding:")
print(f"New shape: {df.shape}")

# === CODE BLOCK SEPARATOR ===

#  بناء smaller_df للنمذجة
print("\n=== Preparing Data for Modeling ===")

drop_cols = ['description', 'day', 'month', 'quarter',
             'city', 'company', 'position', 'industry_cat', 'job_functions',
             'linkedin_id', 'position_id', 'date', 'location', 'salary_mentioned', 'skills']

# Only drop columns that exist in the dataframe
existing_drop_cols = [col for col in drop_cols if col in df.columns]
print(f"Dropping columns: {existing_drop_cols}")

smaller_df = df.drop(existing_drop_cols, axis=1)
print("Smaller DataFrame:")
print(f"Shape: {smaller_df.shape}")
print(f"Columns: {smaller_df.columns.tolist()}")

# === CODE BLOCK SEPARATOR ===

#  تقسيم البيانات + Scaling
print("\n=== Data Splitting ===")

X = smaller_df.drop('level', axis=1)
y = smaller_df['level']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# Stratify للمحافظة على توزيع الفئات
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# === CODE BLOCK SEPARATOR ===

# Scaling
print("\n=== Feature Scaling ===")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("Data scaled successfully!")

# === CODE BLOCK SEPARATOR ===

# Logistic Regression
print("\n=== Training Baseline Models ===")
print("\n--- Logistic Regression ---")
log_clf = LogisticRegression(random_state=42, max_iter=1000)
log_clf.fit(X_train_scaled, y_train)

y_pred_log = log_clf.predict(X_test_scaled)

print("Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

# === CODE BLOCK SEPARATOR ===

print("\n--- Random Forest ---")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_scaled, y_train)

y_pred_rf = rf_clf.predict(X_test_scaled)

print("Random Forest (Baseline) Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# === CODE BLOCK SEPARATOR ===

print("\n--- XGBoost ---")
xgb_clf = XGBClassifier(
    random_state=42,
    n_estimators=200,
    n_jobs=-1,
    objective='multi:softmax'
)

xgb_clf.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_clf.predict(X_test_scaled)

print("XGBoost (Baseline) Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))

# === CODE BLOCK SEPARATOR ===

print("\n--- KNN ---")
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train_scaled, y_train)
y_pred_knn = knn_clf.predict(X_test_scaled)

print("KNN (Baseline) Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# === CODE BLOCK SEPARATOR ===

#  معالجة عدم توازن الفئات باستخدام SMOTE
print("\n=== Applying SMOTE for Class Imbalance ===")
print("Before SMOTE:", y_train.value_counts().to_dict())

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

print("After SMOTE:", pd.Series(y_train_res).value_counts().to_dict())
print("X_train_res:", X_train_res.shape)
print("y_train_res:", y_train_res.shape)

# === CODE BLOCK SEPARATOR ===

# Logistic Regression with SMOTE
print("\n=== Training Models with SMOTE ===")
print("\n--- Logistic Regression with SMOTE ---")
log_res = LogisticRegression(random_state=42, max_iter=1000)
log_res.fit(X_train_res, y_train_res)

y_pred_log_res = log_res.predict(X_test_scaled)

print("Logistic Regression after SMOTE:")
print("Accuracy:", accuracy_score(y_test, y_pred_log_res))
print(classification_report(y_test, y_pred_log_res))

# === CODE BLOCK SEPARATOR ===

#Random Forest with SMOTE
print("\n--- Random Forest with SMOTE ---")
rf_res = RandomForestClassifier(
    n_estimators=200,
    criterion='entropy',
    random_state=42,
    n_jobs=-1
)
rf_res.fit(X_train_res, y_train_res)

y_pred_rf_res = rf_res.predict(X_test_scaled)

print("Random Forest after SMOTE:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf_res))
print(classification_report(y_test, y_pred_rf_res))

# === CODE BLOCK SEPARATOR ===

#XGBoost with SMOTE
print("\n--- XGBoost with SMOTE ---")
xgb_res = XGBClassifier(
    random_state=42,
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    objective='multi:softmax'
)

xgb_res.fit(X_train_res, y_train_res)
y_pred_xgb_res = xgb_res.predict(X_test_scaled)

print("XGBoost after SMOTE:")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb_res))
print(classification_report(y_test, y_pred_xgb_res))

# === CODE BLOCK SEPARATOR ===

#KNN with SMOTE
print("\n--- KNN with SMOTE ---")
knn_res = KNeighborsClassifier(n_neighbors=27)
knn_res.fit(X_train_res, y_train_res)

y_pred_knn_res = knn_res.predict(X_test_scaled)

print("KNN after SMOTE:")
print("Accuracy:", accuracy_score(y_test, y_pred_knn_res))
print(classification_report(y_test, y_pred_knn_res))

# === CODE BLOCK SEPARATOR ===

# حفظ الـ scaler
print("\n=== Saving Models ===")
joblib.dump(scaler, "scaler_jobs_level.pkl")

# حفظ مودل Random Forest النهائي
joblib.dump(rf_res, "Random_Forest_level_model.pkl")

print("Saved: scaler_jobs_level.pkl and Random_Forest_level_model.pkl")

# === CODE BLOCK SEPARATOR ===

level_mapping = {
    0: "Not Applicable",
    1: "Internship",
    2: "Entry level",
    3: "Associate",
    4: "Mid-Senior level",
    5: "Director",
    6: "Executive"
}

with open("level_mapping.json", "w", encoding='utf-8') as f:
    json.dump(level_mapping, f, ensure_ascii=False)

print("Saved: level_mapping.json")

# === CODE BLOCK SEPARATOR ===

smaller_df.to_csv("smaller_df_prepared.csv", index=False)
print("Saved: smaller_df_prepared.csv")

# === CODE BLOCK SEPARATOR ===

# Prepare Train/Test Split for NLP Using Same Indices
print("\n=== NLP Processing ===")
# نستخدم نفس الـ index المستخدم في smaller_df split
train_idx = X_train.index
test_idx  = X_test.index

# نتأكد أن df و smaller_df على نفس الفهارس
assert all(train_idx.isin(df.index)), "Mismatch between df and smaller_df indices"

# نصوص الوصف
X_train_text = df.loc[train_idx, 'description'].astype(str)
X_test_text  = df.loc[test_idx, 'description'].astype(str)

print("Train text shape:", X_train_text.shape)
print("Test text shape :", X_test_text.shape)

# === CODE BLOCK SEPARATOR ===

#TF-IDF Vectorization (NLP Features)
print("\n--- TF-IDF Vectorization ---")

tfidf = TfidfVectorizer(
    max_features=10000,   # عدد أقصى للـ features لتقليل الأبعاد
    ngram_range=(1, 2),   # Unigram + Bigram
    min_df=5              # الكلمة تظهر في 5 وثائق على الأقل
)

X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf  = tfidf.transform(X_test_text)

print("X_train_tfidf shape:", X_train_tfidf.shape)
print("X_test_tfidf  shape:", X_test_tfidf.shape)

# === CODE BLOCK SEPARATOR ===

#  NLP Model: TF-IDF + Logistic Regression
print("\n--- NLP Model Training ---")

nlp_clf = LogisticRegression(
    max_iter=1000,
    multi_class='multinomial',
    n_jobs=-1
)

nlp_clf.fit(X_train_tfidf, y_train)

y_pred_nlp = nlp_clf.predict(X_test_tfidf)

print("NLP Model (TF-IDF + Logistic Regression) Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_nlp))
print(classification_report(y_test, y_pred_nlp))

cm_nlp = confusion_matrix(y_test, y_pred_nlp)
print("Confusion Matrix (NLP):")
print(cm_nlp)

# === CODE BLOCK SEPARATOR ===

#  Simple Ensemble : Tabular RF + NLP
print("\n=== Ensemble Model ===")
# احتمالات من RF (Tabular)
proba_rf = rf_res.predict_proba(X_test_scaled) 

# احتمالات من NLP Model
proba_nlp = nlp_clf.predict_proba(X_test_tfidf)

# مزج بسيط: متوسط الاحتمالات
alpha = 0.5  # وزن RF
beta  = 0.5  # وزن NLP
proba_ens = alpha * proba_rf + beta * proba_nlp

y_pred_ens = proba_ens.argmax(axis=1)

print("Ensemble (RF + NLP) Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_ens))
print(classification_report(y_test, y_pred_ens))

# === CODE BLOCK SEPARATOR ===

#  Save NLP Artifacts (TF-IDF + NLP Model)
joblib.dump(tfidf, "tfidf_description.pkl")
joblib.dump(nlp_clf, "nlp_level_model.pkl")

print("Saved: tfidf_description.pkl, nlp_level_model.pkl")

# === CODE BLOCK SEPARATOR ===

# Model Performance Summary
print("\n" + "="*60)
print("MODEL PERFORMANCE SUMMARY")
print("="*60)

models_performance = {
    "Logistic Regression (Baseline)": accuracy_score(y_test, y_pred_log),
    "Random Forest (Baseline)": accuracy_score(y_test, y_pred_rf),
    "XGBoost (Baseline)": accuracy_score(y_test, y_pred_xgb),
    "KNN (Baseline)": accuracy_score(y_test, y_pred_knn),
    "Logistic Regression + SMOTE": accuracy_score(y_test, y_pred_log_res),
    "Random Forest + SMOTE": accuracy_score(y_test, y_pred_rf_res),
    "XGBoost + SMOTE": accuracy_score(y_test, y_pred_xgb_res),
    "KNN + SMOTE": accuracy_score(y_test, y_pred_knn_res),
    "NLP Model (TF-IDF + LR)": accuracy_score(y_test, y_pred_nlp),
    "Ensemble (RF + NLP)": accuracy_score(y_test, y_pred_ens)
}

for model, accuracy in sorted(models_performance.items(), key=lambda x: x[1], reverse=True):
    print(f"{model:<30}: {accuracy:.4f}")

print("\nScript completed successfully!")
print("All models trained and saved!")