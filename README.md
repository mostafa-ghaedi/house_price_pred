# 🏠 House Price Prediction Project

## Overview
This project predicts house sale prices using a Ridge Regression model based on the House Prices dataset from Kaggle.

## How to Run
1. Activate virtual environment:
   ```
   .\venv\Scripts\activate
   ```
2. Install dependencies:
   ```
   pip install pandas numpy scikit-learn matplotlib streamlit
   ```
3. Run the dashboard:
   ```
   streamlit run dashboard.py
   ```
   - Place `train.csv` in the project folder.

## Features
- Filters for Year Built, Lot Area, and Overall Quality.
- Displays MSE, R², feature importance, and actual vs predicted prices.

## Results
- MSE: 1302988439.71
- R²:  0.8301

## Lessons Learned
- Preprocessing numeric and categorical features is essential.
- Ridge Regression helps with overfitting on complex datasets.

## Future Improvements
- Add more features (e.g., GarageArea, TotalBsmtSF).
- Implement cross-validation.
- Try other models (e.g., Random Forest).