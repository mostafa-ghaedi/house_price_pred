# 🏠 House Price Prediction with Random Forest

## Overview
This project predicts house sale prices using a **Random Forest Regressor** model based on the [House Prices dataset from Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). The project includes data preprocessing, model training, evaluation, and an interactive dashboard built with Streamlit. The Random Forest model was chosen for its ability to handle complex relationships in the data and reduce overfitting through ensemble learning.

### Key Features
- **Data Preprocessing**: Handles missing values, encodes categorical variables, and scales numerical features.
- **Model Training**: Utilizes a Random Forest Regressor with 100 trees for robust predictions.
- **Evaluation**: Includes metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), R² Score, and Cross-Validated MSE.
- **Feature Importance**: Displays the top 10 features influencing house prices.
- **Interactive Dashboard**: Allows users to filter data and view model performance dynamically.

## Setup

### Prerequisites
- Python 3.8 or later
- A virtual environment (recommended)
- Git (for version control)

### Installation
1. **Clone the Repository**:
   ```
   git clone <your-repo-url>
   cd house-price-prediction
   ```

2. **Create and Activate a Virtual Environment**:
   ```
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   # source venv/bin/activate  # On Linux/Mac
   ```

3. **Install Dependencies**:
   ```
   pip install pandas numpy scikit-learn matplotlib streamlit
   ```

4. **Download the Dataset**:
   - Place `train.csv` from the [Kaggle House Prices dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) in the project root.

## Usage

### 1. Preprocess the Data
Run the preprocessing script to handle missing values, encode categorical features, and scale numerical features:
```
python preprocess.py
```
This generates `X_processed.npy`, `y.npy`, and `feature_names.npy`.

### 2. Train the Random Forest Model
Train the model and evaluate its performance:
```
python model_rf.py
```
This will output the MSE, MAE, R² Score, Cross-Validated MSE, and the top 10 most important features.

### 3. Launch the Interactive Dashboard
Explore the model's predictions and feature importance interactively:
```
streamlit run dashboard_rf.py
```
- Use the sidebar filters to adjust the data range for Year Built, Lot Area, and Overall Quality.
- View the filtered dataset, model performance metrics, feature importance, and a scatter plot of actual vs. predicted prices.

## Results
- **Overall Model Performance**:
  - Mean Squared Error (MSE): 840188622.69
  - Mean Absolute Error (MAE): 19224.91
  - R² Score: 0.8905
  - Cross-Validated MSE: 994475654.91
- **Top Features**: The model identifies key features influencing house prices, such as `OverallQual`, `GrLivArea`, and `Neighborhood`.

## Lessons Learned
- **Random Forest Benefits**: Outperforms simpler models like Ridge Regression by capturing non-linear relationships.
- **Feature Importance**: Provides insight into which features most affect predictions, aiding in model interpretability.
- **Cross-Validation**: Ensures the model's robustness by evaluating it on multiple data splits.

## Future Improvements
- **Hyperparameter Tuning**: Optimize `n_estimators`, `max_depth`, etc., for better performance.
- **Additional Features**: Incorporate more dataset columns (e.g., `GarageArea`, `TotalBsmtSF`).
- **Model Comparison**: Test other ensemble methods like XGBoost or Gradient Boosting.
- **Enhanced Visualizations**: Add interactive plots using Plotly for a richer dashboard experience.

## Project Structure
- `preprocess.py`: Handles data cleaning and preprocessing.
- `model_rf.py`: Trains and evaluates the Random Forest model.
- `dashboard_rf.py`: Launches the Streamlit dashboard for interactive exploration.
- `.gitignore`: Excludes virtual environment and large files from version control.
- `README.md`: This documentation file.

## Contact
For questions or feedback, please reach out via GitHub or email at [mustafa.ghaedi@gmail.com].
