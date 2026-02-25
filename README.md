# 🚀 RetentionAI — Customer Churn Prediction
> An end-to-end machine learning web application that predicts customer churn in real time using a **Random Forest classifier** (93.7% accuracy, AUC 0.95). Built with Python, scikit-learn, and Streamlit.

---

## 🔍 What is this project?

**RetentionAI** helps businesses identify customers who are likely to cancel their subscription (churn) before it happens. The app:

- Takes customer profile data as input (demographics, usage, billing, etc.)
- Runs it through a trained Random Forest model
- Returns a **real-time churn risk score** with a confidence percentage
- Supports **batch processing** — upload a CSV and score thousands of customers at once
- Includes a **Risk Analysis** dashboard with model performance charts

---

## 🗂️ Project Structure

```
Churn/
│
├── churn_app.py                              ← Streamlit web app (main entry point)
├── train_model.py                            ← Standalone model training script
├── Churn_Prediction_Final.ipynb              ← EDA & experiment notebook
│
├── customer_churn_dataset-training-master.csv  ← Training dataset (~380k rows)
├── customer_churn_dataset-testing-master.csv   ← Testing dataset (~127k rows)
│
├── churn_model.joblib                        ← Trained Random Forest model (auto-generated)
├── scaler.joblib                             ← StandardScaler artifact (auto-generated)
├── le_gender.joblib                          ← Gender label encoder (auto-generated)
├── le_subscription.joblib                    ← Subscription encoder (auto-generated)
├── le_contract.joblib                        ← Contract encoder (auto-generated)
├── feature_names.joblib                      ← Feature column order (auto-generated)
│
├── model_report.txt                          ← Full training evaluation report
├── workflow_guide.html                       ← Visual step-by-step guide
├── requirements.txt                          ← Python dependencies
└── README.md                                 ← You are here
```

---

## 📊 Dataset

The datasets used are from Kaggle's **Customer Churn Dataset**:

> Both CSVs are **combined and re-split** (70/15/15) before training to fix distribution mismatch.

**Features used (10 total):**

| Feature | Type | Description |
|---------|------|-------------|
| Age | Numeric | Customer age (18–65) |
| Gender | Categorical | Male / Female |
| Tenure | Numeric | Months as customer (1–60) |
| Usage Frequency | Numeric | Monthly service usage count |
| Support Calls | Numeric | # of support calls made |
| Payment Delay | Numeric | Days of payment delay |
| Subscription Type | Categorical | Basic / Standard / Premium |
| Contract Length | Categorical | Monthly / Quarterly / Annual |
| Total Spend | Numeric | Cumulative spend ($) |
| Last Interaction | Numeric | Days since last contact |

---

## 🤖 Model Performance

Five models were trained and evaluated. **Random Forest was selected as the best model.**

| Model | Train Acc | Val Acc | Test Acc | Val AUC |
|-------|-----------|---------|----------|---------|
| **Random Forest** ✅ | 93.75% | 93.58% | **93.71%** | **0.953** |
| Decision Tree | 93.48% | 93.31% | 93.45% | 0.952 |
| Gradient Boosting | 93.22% | 93.16% | 93.29% | 0.953 |
| Logistic Regression | 83.33% | 83.37% | 83.53% | 0.900 |
| Linear SVM | 83.32% | 83.36% | 83.54% | 0.900 |

**Overfitting gap: 0.0017** (extremely low — model generalizes very well)

---

## ⚙️ How It Works

```
Raw CSVs (train + test)
        ↓
   Combined & Cleaned
        ↓
  Stratified 70/15/15 Split
        ↓
  StandardScaler (fit on train only)
        ↓
  5 Models Trained & Compared
        ↓
  Best Model Saved (Random Forest)
        ↓
  Streamlit App Loads Artifacts → Real-time Predictions
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/RetentionAI-Churn-Prediction.git
cd RetentionAI-Churn-Prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the datasets

Download the datasets from Kaggle (link below) and place both CSV files in the project root:

```
customer_churn_dataset-training-master.csv
customer_churn_dataset-testing-master.csv
```

> 📎 Dataset source: [Kaggle — Customer Churn Dataset](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset)

### 4. (Optional) Re-train the model

```bash
python train_model.py
```

This will generate all `.joblib` artifacts and `model_report.txt`. Skip this step if the `.joblib` files are already present.

### 5. Launch the app

```bash
streamlit run churn_app.py
```

Open your browser at `http://localhost:8501` 🎉

---

## 🧭 App Features

### 🎯 Individual Prediction
Fill in a customer's profile using the sliders and dropdowns to get an instant churn risk score with probability.

### 📊 Risk Analysis
View model performance charts — ROC curve, confusion matrix, feature importances, and AUC scores.

### 📁 Batch Processing
Upload a CSV file with multiple customers and download predictions for all of them at once.

### 📖 How It Works
A built-in explainer tab walks through the ML pipeline step by step.

---

## 📦 About the `.joblib` Files

The `.joblib` files are **binary serialized ML artifacts** — they look empty in a text editor, which is completely normal. They store:

- The trained Random Forest model (~23 MB of compressed decision trees)
- The fitted StandardScaler (to apply the same normalization at inference)
- Three LabelEncoders (for Gender, Subscription Type, Contract Length)
- The ordered feature names list

The app auto-trains and saves these on first run (~30 seconds) if they're missing.

---

## 📁 Datasets on GitHub

Because the raw CSV files are large, they are uploaded as `.zip` files to keep the repository size manageable. The app and training script both read `.zip` files directly via `pandas.read_csv()` — **no manual unzipping needed**.

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **pandas** — data loading and preprocessing
- **scikit-learn** — ML models, scaling, encoding, metrics
- **Streamlit** — web app UI
- **matplotlib** — charts and visualizations
- **joblib** — model serialization

---

## 📄 License

This project is licensed under the MIT License — feel free to use, modify, and distribute.

---
