# 🩺 Diabetes Prediction using Machine Learning (Healthcare Domain)

A complete **end-to-end machine learning project** for predicting diabetes risk using real-world clinical features.  
This project covers everything from **EDA → preprocessing → multi-model training → hyperparameter tuning → explainability (SHAP) → Streamlit deployment**.

---

## 🚀 Live Demo (Streamlit App)
Users can input patient details and get:
- ✅ Diabetes probability
- ✅ Risk category (Low / Moderate / High)
- ✅ Visual risk gauge
- ✅ Feature snapshot

---

## 📌 Problem Statement
Diabetes is one of the fastest-growing lifestyle diseases worldwide. Early detection can help prevent severe complications such as heart disease, kidney failure, and nerve damage.

This project aims to:
> **Build a reliable and interpretable ML model to predict diabetes risk using clinical and demographic data.**

---

## 🗂 Dataset Overview
- 📦 Source: Kaggle
- 🔢 Total Records: ~100,000
- 🎯 Target Variable: `diabetes` (0 = No, 1 = Yes)

---
## 🧰 Tech Stack & Tools

### Programming & Data Handling  
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)  
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)  
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

### Machine Learning & Modeling  
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)  
![XGBoost](https://img.shields.io/badge/XGBoost-EC4E20?style=for-the-badge&logo=xgboost&logoColor=white)  
![Imbalanced Learn](https://img.shields.io/badge/SMOTE-Imbalanced--Learn-blue?style=for-the-badge)

### Explainable AI  
![SHAP](https://img.shields.io/badge/SHAP-Explainable--AI-brightgreen?style=for-the-badge)

### Visualization  
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)  
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge)  
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

### Deployment  
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

### Model Storage & Utilities  
![Joblib](https://img.shields.io/badge/Joblib-9C27B0?style=for-the-badge)  
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)

---

### Key Features:
- `age`
- `gender`
- `hypertension`
- `heart_disease`
- `bmi`
- `HbA1c_level`
- `blood_glucose_level`
- `smoking_history`

---

## 🔍 Exploratory Data Analysis (EDA)
Performed detailed analysis including:
- ✅ Target class imbalance check
- ✅ Age-wise diabetes trend
- ✅ BMI, HbA1c, and glucose distributions
- ✅ Smoking vs diabetes patterns
- ✅ Hypertension & heart disease interaction
- ✅ Correlation heatmap
- ✅ Outlier analysis (IQR method)
- ✅ Skewness & distribution checks

📌 **Key Insight:**  
HbA1c and blood glucose are the strongest predictors of diabetes.

---

## ⚙️ Preprocessing Pipeline
- ✅ Numerical Features → **RobustScaler**
- ✅ Categorical Features → **OneHotEncoder**
- ✅ Class Imbalance → **SMOTE applied on training data only**
- ✅ Train-Test Split → **Stratified**

---

## 🤖 Model Training & Benchmarking

| Model | Recall | Precision | F1 | ROC-AUC |
|------|--------|-----------|----|----------|
| Logistic Regression | 0.89 | 0.43 | 0.57 | 0.96 |
| Random Forest | 0.76 | 0.75 | 0.75 | 0.97 |
| Gradient Boosting | 0.78 | 0.73 | 0.76 | 0.97 |
| ✅ **XGBoost (Final Model)** | 0.71 | 0.89 | 0.79 | **0.977** |

📌 **Final Model Selected:** `XGBoost`  
Selected based on best **F1-score + ROC-AUC + precision-recall balance**.

---

## ✅ Final Model Performance (Test Set)
- ✅ Accuracy: **97%**
- ✅ ROC–AUC: **0.978**
- ✅ Precision (Diabetes): **91%**
- ✅ Recall (Diabetes): **71%**

---

## 🔎 Model Explainability (SHAP)
- ✅ Global feature importance (summary + bar plots)
- ✅ Individual patient explanations (waterfall plots)
- ✅ Clinically meaningful patterns:
  - HbA1c ↑ → Risk ↑
  - Glucose ↑ → Risk ↑
  - BMI & Age → Moderate influence

---

## 🖥 Deployment (Streamlit)
Features of the web app:
- ✅ Interactive UI
- ✅ Live probability prediction
- ✅ Risk gauge meter
- ✅ Feature visualization
- ✅ Fast & lightweight

---
## 🧠 Key Learnings

- Handling real healthcare imbalance
- Robust feature scaling
- Multi-model benchmarking
- Explainable AI using SHAP
- End-to-end ML deployment

---
## ⚠ Disclaimer

- This project is for educational and research purposes only and should not be used as a substitute for professional medical diagnosis.

---
## 👨‍💻 Author

- Abdul Qadir \
- BS Applied AI & Data Science \
- Healthcare ML | Data Science | Explainable AI