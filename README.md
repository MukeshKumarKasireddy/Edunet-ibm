# Edunet-ibm
# 💼 Employee Salary Prediction using Machine Learning

This project predicts the estimated salary of an employee based on their job title, education, department, location, work type, job type, and experience level.  
It uses **Random Forest Regressor** and provides an interactive web app using **Streamlit**.

---

## 🔍 Problem Statement

Companies often struggle to set fair and consistent salary ranges. This app uses machine learning to estimate salaries based on employee profiles, making the process more efficient and data-driven.

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **Scikit-learn** – model training and evaluation
- **Pandas / NumPy** – data handling
- **Matplotlib / Seaborn** – visualizations
- **Streamlit** – frontend web interface

---

## 🚀 Features

- Live salary prediction with user inputs
- Easy-to-use Streamlit web interface
- Dropdowns and sliders for clean UX
- Salary distribution visualization
- Built-in model training with real-time results
- Runs locally on your browser (`localhost:8501`)

---

## 🎯 How It Works

1. Generates synthetic dataset with 1000 records
2. Label encodes categorical features
3. Splits data into training and testing sets
4. Trains a Random Forest Regressor
5. Displays R² score to evaluate accuracy
6. Web app accepts user input and shows predicted salary
7. Histogram shows where the prediction lies in dataset

---

## 🧠 Machine Learning Model

- **Model Used**: `RandomForestRegressor`
- **Reason**: Outperformed other models (like Linear Regression, Gradient Boosting) with an R² score of ~0.89
- **Performance Metric**: R² Score

---

## 📁 Project Structure

```bash
├── app.py                  # Streamlit app file
|── README.md               # Project documentation
