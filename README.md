# 🏥 Health Insurance Cost Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Best%20Model-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-EC2-232F3E?style=for-the-badge&logo=amazonaws&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-2EA44F?style=for-the-badge)

**Predict individual medical insurance costs using demographic and lifestyle data.**

</div>

---

## 📸 Preview

| Prediction Output |
|:-:|
| ![Prediction Page](images/prediction.png) |

---

## 🎯 Problem Statement

Medical insurance costs vary significantly across individuals based on factors like age, BMI, and lifestyle habits. This project builds a machine learning model to **predict insurance charges** based on a person's demographic and health profile — enabling faster, data-driven cost estimation.

---

## 📊 Dataset Features

| Feature | Description |
|---|---|
| `age` | Age of the individual |
| `gender` | Male / Female |
| `bmi` | Body Mass Index |
| `children` | Number of dependents |
| `smoker` | Smoking status (`yes` / `no`) |
| `region` | Residential area |
| `charges` | 💰 Insurance cost *(target variable)* |

> **Key insight:** Smoking status and BMI were found to be the strongest predictors of insurance charges.

---

## ⚙️ Workflow

### 1. 🔍 Exploratory Data Analysis
- Visualized feature distributions and correlations
- Identified `smoker` and `bmi` as dominant factors affecting charges

### 2. 🛠️ Preprocessing
- Label encoding for categorical variables (`gender`, `smoker`, `region`)
- Feature scaling with `StandardScaler`

### 3. 🤖 Model Training

| Model | Notes |
|---|---|
| Linear Regression | Baseline |
| Random Forest | Better with non-linearity |
| **XGBoost** | ✅ **Best performance** |

### 4. 📈 Evaluation

| Metric | Score |
|---|---|
| R² Score | ~0.80 |
| MAE | Evaluated |
| RMSE | Evaluated |

---

## 🚀 Deployment

The app is deployed as an interactive **Streamlit** web app where users enter their details and get an instant cost prediction.

### Run Locally

```bash
# Clone the repository
git clone https://github.com/your-username/health-insurance-predictor.git
cd health-insurance-predictor

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

---

## 📁 Project Structure

```
health-insurance-predictor/
│
├── app.py                  # Streamlit app
├── model/
│   └── xgboost_model.pkl   # Trained model
├── notebooks/
│   └── EDA_and_Training.ipynb
├── images/
│   └── prediction.png
├── requirements.txt
└── README.md
```

---

## 🧰 Tech Stack

- **Language:** Python 3.10
- **ML Libraries:** Scikit-learn, XGBoost
- **Web App:** Streamlit
- **Deployment:** AWS EC2

---

## 🙌 Acknowledgements

Dataset sourced from the [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) on Kaggle.