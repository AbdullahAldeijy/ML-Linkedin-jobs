# LinkedIn Jobs Level Prediction - Machine Learning Pipeline

A comprehensive machine learning project for predicting job levels from LinkedIn job postings using various classification algorithms and feature engineering techniques.

## 📊 Project Overview

This project analyzes LinkedIn job data to predict job levels (Entry level, Associate, Mid-Senior level, Director, Executive, etc.) based on job characteristics such as region, industry, experience requirements, and job descriptions.

## 🎯 Key Features

- **Multi-class Classification**: Predicts 7 different job levels
- **Feature Engineering**: Advanced preprocessing including one-hot encoding and text vectorization
- **Class Imbalance Handling**: SMOTE implementation for balanced training
- **Multiple Algorithms**: Comparison of various ML models
- **NLP Integration**: TF-IDF vectorization of job descriptions
- **Model Ensemble**: Combines tabular and text-based models

## 📈 Model Performance

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 35.94% |
| **Random Forest** | **54.06%** |
| Random Forest + SMOTE | 51.59% |

## 🗂️ Dataset

- **Size**: 41,222 job records
- **Features**: 21 original columns including job descriptions, regions, industries, experience levels
- **Target**: Job level classification (7 classes)
- **Source**: LinkedIn job postings

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Download Large Files
```bash
python download_data.py
```

### Fast Execution
```bash
python FeatureEngineeringandML_fast.py
```

### Full Pipeline (Comprehensive)
```bash
python FeatureEngineeringandML_corrected.py
```

## 📁 File Structure

```
├── data_jobs.csv                           # Dataset
├── FeatureEngineeringandML_fast.py         # Quick ML pipeline
├── FeatureEngineeringandML_corrected.py    # Full ML pipeline
├── Random_Forest_level_model.pkl           # Trained model
├── scaler_jobs_level.pkl                   # Feature scaler
├── level_mapping.json                      # Job level mappings
├── feature_info.json                       # Feature information
└── README.md                               # This file
```

## 🔧 Technical Details

### Feature Engineering
- **Region Encoding**: Label encoding for 111 different regions
- **Industry Encoding**: One-hot encoding for top industries
- **Text Processing**: TF-IDF vectorization of job descriptions
- **Numerical Features**: Experience years, description length, remote work flag

### Models Implemented
1. **Logistic Regression**
2. **Random Forest** (Best performer)
3. **XGBoost**
4. **K-Nearest Neighbors**
5. **NLP Model** (TF-IDF + Logistic Regression)
6. **Ensemble Model** (RF + NLP)

### Class Distribution
- Mid-Senior level: 29.60%
- Entry level: 26.39%
- Associate: 25.53%
- Not Applicable: 10.08%
- Director: 5.56%
- Executive: 1.93%
- Internship: 0.92%

## 📊 Results

The Random Forest model achieved the best performance with **54.06% accuracy** on the test set. Given the complexity of predicting job levels across 7 different categories, this represents solid performance for practical applications.

### Classification Report (Best Model)
```
              precision    recall  f1-score   support
           0       0.52      0.66      0.58      1246
           1       0.49      0.93      0.64       113
           2       0.60      0.53      0.56      3264
           3       0.54      0.47      0.50      3157
           4       0.59      0.52      0.55      3661
           5       0.22      0.36      0.27       687
           6       0.17      0.40      0.24       239

    accuracy                           0.52     12367
   macro avg       0.45      0.55      0.48     12367
weighted avg       0.54      0.52      0.52     12367
```

## 🛠️ Usage

### Loading the Trained Model
```python
import joblib
import json

# Load model and scaler
model = joblib.load('Random_Forest_level_model.pkl')
scaler = joblib.load('scaler_jobs_level.pkl')

# Load mappings
with open('level_mapping.json', 'r') as f:
    level_mapping = json.load(f)

# Make predictions
predictions = model.predict(scaled_features)
predicted_levels = [level_mapping[str(pred)] for pred in predictions]
```

## 📋 Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- xgboost
- imbalanced-learn
- matplotlib
- seaborn
- joblib

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements.

## 📄 License

This project is open source and available under the MIT License.

## 📧 Contact

For questions or collaboration opportunities, please reach out through GitHub issues.

---

**Note**: The full pipeline (`FeatureEngineeringandML_corrected.py`) includes comprehensive visualizations and detailed analysis but takes longer to run. Use the fast version (`FeatureEngineeringandML_fast.py`) for quick results.