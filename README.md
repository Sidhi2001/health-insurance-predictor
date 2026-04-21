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

Five regression models were trained and compared:

| Model | Best Parameters |
|---|---|
| Linear Regression | — |
| Polynomial Regression (deg 2) | — |
| Random Forest | `max_depth=10`, `n_estimators=100`, `min_samples_split=2`, `min_samples_leaf=2` |
| SVR | `C=50`, `kernel=linear`, `epsilon=0.1`, `degree=2` |
| **XGBoost** | `learning_rate=0.05`, `max_depth=3`, `n_estimators=100`, `subsample=1.0` |

### 4. 📈 Model Evaluation

| Model | R² Score | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.6545 | 5181.15 | 6895.40 |
| Polynomial Regression (deg 2) | 0.7494 | 4321.34 | 5872.80 |
| Random Forest | 0.7857 | 4134.58 | 5430.40 |
| SVR | 0.4372 | 5582.50 | 8800.55 |
| **XGBoost** ✅ | **0.7971** | **3968.18** | **5283.97** |

> **XGBoost** achieved the best performance with the highest R² score and lowest MAE & RMSE, and was selected as the final model.

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

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML & Preprocessing | Scikit-learn, XGBoost |
| Encoding | LabelEncoder, StandardScaler |
| Notebook | Jupyter Notebook |
| Web App | Streamlit |
| Deployment | AWS EC2 |

---

## 🙌 Acknowledgements

Dataset sourced from the [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) on Kaggle.


AWS : 

http://16.171.37.12:8501/