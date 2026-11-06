# ♻️ Smart Waste Generation Prediction

### Machine Learning-Based Waste Analytics and Prediction

## 📌 Project Overview

Smart Waste Generation Prediction is a machine learning project designed to predict daily waste generation using demographic, environmental, and waste-management-related factors.

The project analyses patterns in waste generation across different cities and wards to understand how population, households, weather conditions, commercial activity, tourism, and other factors influence waste production.

It aims to support data-driven waste management, better resource allocation, and sustainable waste prevention strategies.

## 🎯 Objectives

- Predict daily waste generation using regression algorithms.
- Analyse factors influencing waste production.
- Perform Exploratory Data Analysis (EDA) to identify patterns and trends.
- Compare multiple regression models to identify the best-performing model.
- Apply statistical techniques to understand the dataset.
- Present predictions and analytical insights through an interactive Streamlit dashboard.

## 📊 Dataset

- **Dataset Size:** 50,000 records
- **Number of Columns:** 16
- **Data Type:** Synthetic dataset
- **Target Variable:** `waste_generated_tonnes_perday`

### Features Used

| Feature | Description |
|---|---|
| City | City name |
| Ward | Ward identification |
| Waste Type | Category of waste |
| Population | Population of the area |
| Households | Number of households |
| Temperature | Temperature in Celsius |
| Rainfall | Rainfall in mm |
| Humidity | Humidity percentage |
| Commercial Activity Index | Level of commercial activity |
| Tourism Index | Tourism activity level |
| Festival Flag | Indicates festival periods |
| Day of Week | Day of the week |
| Weekend Flag | Indicates weekends |

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab
- Streamlit

## 🤖 Machine Learning Algorithms

The following regression algorithms were implemented and compared:

1. Linear Regression
2. Decision Tree Regression
3. Ridge Regression
4. Lasso Regression
5. Random Forest Regression

### Model Evaluation Metrics

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

### Best Performing Model

**Random Forest Regressor**

| Evaluation Metric | Result |
|---|---:|
| R² Score | 0.8749 |
| MAE | 0.4299 |
| RMSE | 0.7007 |

Random Forest was selected based on its predictive performance among the evaluated regression models.

## 📈 Statistical Analysis

The project also incorporates statistical analysis techniques:

- Exploratory Data Analysis
- Sampling Techniques
- Central Limit Theorem (CLT)
- Bootstrap Sampling
- 95% Confidence Intervals
- Hypothesis Testing using t-tests
- ANOVA
- Chi-Square Test

## 🔄 Project Workflow

1. Synthetic dataset generation and loading
2. Data cleaning and preprocessing
3. Exploratory Data Analysis
4. Feature selection
5. Categorical encoding using One-Hot Encoding
6. Numerical feature scaling using StandardScaler
7. Train-test split (80:20)
8. Regression model training
9. Model evaluation and comparison
10. Statistical analysis
11. Selection of the best-performing model
12. Interactive visualization using Streamlit

## 🖥️ Application Interface

<img width="1459" height="834" alt="image" src="https://github.com/user-attachments/assets/50e4fdff-d45e-45d8-a80c-c3028e97ad41" />


The project uses Streamlit to provide an interactive frontend for presenting waste generation predictions and analytical insights.

Google Colab is used for backend model development and training.

## 🚀 Future Scope

- Integration with real-time municipal waste collection data.
- Expansion to more cities and geographical regions.
- Improved forecasting using real-world datasets.
- Waste reduction and prevention recommendations.
- Integration with smart city waste management systems.

## 👩‍💻 Project Details

**Project Name:** Smart Waste Generation Prediction

**Domain:** Machine Learning | Regression | Data Analytics

**Developed by:** Vedika Kulkarni


## 📌 Conclusion

This project demonstrates how machine learning and statistical analysis can be applied to understand and predict waste generation patterns. By identifying the factors affecting waste production and providing data-driven predictions, the system aims to contribute towards more efficient and sustainable waste management.
