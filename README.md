# 🚢 Titanic Survival Predictor

🚀 **Try the Titanic Survival Predictor:**
https://titanic-survival-predictor-8lxahcnwblbgmoyetacarm.streamlit.app/


## 📌 About the Project

The **Titanic Survival Predictor** is a machine-learning project based on the famous Titanic dataset. The goal of this project is to predict passenger survival using historical passenger information.

Users can enter details such as passenger class, gender, age, family information, fare, and embarkation point. The trained machine-learning model processes these inputs and predicts whether the passenger is likely to have **Survived** or **Not Survived**.

This project demonstrates the complete machine-learning workflow, including **data preprocessing, feature selection, model training, prediction, and deployment using Streamlit**.


## ✨ Features

* 🚢 Predicts Titanic passenger survival
* 👤 Accepts passenger information through an interactive interface
* 🤖 Uses a trained machine-learning model
* ⚡ Provides predictions instantly
* 🌐 Deployed online using Streamlit


## 📊 Input Features

The application uses several passenger attributes for prediction:

| Feature      | Description                                |
| ------------ | ------------------------------------------ |
| **Pclass**   | Passenger's ticket class: 1st, 2nd, or 3rd |
| **Sex**      | Passenger's gender                         |
| **Age**      | Passenger's age                            |
| **SibSp**    | Number of siblings or spouses aboard       |
| **Parch**    | Number of parents or children aboard       |
| **Fare**     | Passenger's ticket fare                    |
| **Embarked** | Port where the passenger boarded           |


## 🔄 How It Works

Passenger Information
        ↓
Data Preprocessing
        ↓
Feature Transformation
        ↓
Trained Machine Learning Model
        ↓
Survival Prediction
        ↓
Survived / Not Survived


## 🛠️ Technologies Used

* **Python** – Programming language
* **Pandas** – Data manipulation and analysis
* **NumPy** – Numerical operations
* **Scikit-learn** – Machine learning and preprocessing
* **Streamlit** – Interactive web application
* **Joblib / Pickle** – Model serialization
* **Matplotlib / Seaborn** – Data visualization and analysis

### File Description

* `app.py` – Main Streamlit application
* `Titanic_Survival_Model.pkl` – Trained machine-learning model
* `Titanic-Dataset.csv` – Titanic dataset
* `requirements.txt` – Required Python libraries
* `README.md` – Project documentation
