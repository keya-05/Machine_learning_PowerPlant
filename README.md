Power Plant Energy Output Prediction

This project uses machine learning regression models to predict the net electrical energy output of a power plant based on several environmental factors. It applies data preprocessing, exploratory data analysis (EDA), and various regression models to evaluate and compare performance.

Dataset

The dataset is from the UCI Machine Learning Repository:  
Combined Cycle Power Plant (CCPP) 
It contains 9,568 data points collected from a power plant over 6 years.

Features:
- Ambient Temperature (°C)
- Ambient Pressure (millibar)
- Relative Humidity (%)
- Exhaust Vacuum (cm Hg)

Target:
- Net hourly electrical energy output (PE) in MW

---

Tools & Technologies

- Python
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Seaborn

---

Key Steps

1. Data Cleaning and Preprocessing
   - Checked for null values
   - Scaled features using StandardScaler

2. Exploratory Data Analysis
   - Correlation heatmaps
   - Pair plots for visualizing relationships

3. Model Training
   - Linear Regression
   - Random Forest Regressor
   - Decision Tree Regressor
   - Compared using R² score and RMSE

4. Model Evaluation
   - Chose the best model based on testing accuracy
   - Visualized actual vs predicted output

---

 Results

| Model                 | R² Score | RMSE |
|----------------------|----------|------|
| Linear Regression     | 0.93     | ~4.5 |
| Random Forest Regressor | 0.96  | ~3.0 |
| Decision Tree Regressor | 0.94  | ~3.7 |

>  Random Forest gave the best results among all tested models.

---

 Future Improvements

- Hyperparameter tuning using GridSearchCV
- Deploying the model using Streamlit for live prediction
- Adding cross-validation for better generalization

---
