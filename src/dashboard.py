import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import matplotlib.pyplot as plt

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("D:/machine_learning/house_price_pred/src/train.csv")
        return df
    except FileNotFoundError:
        st.error("Error: 'train.csv' not found. Please download it from Kaggle and place it in the project folder.")
        st.stop()

df = load_data()

# Select features and target
numeric_features = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd", "OverallQual", "GrLivArea"]
categorical_features = ["Neighborhood", "SaleCondition"]
target = "SalePrice"

# Check if features exist
missing_features = [feat for feat in numeric_features + categorical_features if feat not in df.columns]
if missing_features:
    st.error(f"Error: These features are missing in the dataset: {missing_features}")
    st.stop()

X = df[numeric_features + categorical_features]
y = df[target]

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop="first", sparse_output=False))
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Transform data
try:
    X_processed = preprocessor.fit_transform(X)
except Exception as e:
    st.error(f"Error during preprocessing: {e}")
    st.stop()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Train model
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Streamlit Dashboard
st.title("🏠 House Price Prediction Dashboard")
st.markdown("Predict house prices using a Ridge Regression model with interactive filters.")

# Sidebar for filters
st.sidebar.header("🔍 Filter Options")
year_range = st.sidebar.slider("Year Built Range", int(df["YearBuilt"].min()), int(df["YearBuilt"].max()), (int(df["YearBuilt"].min()), int(df["YearBuilt"].max())))
lotarea_range = st.sidebar.slider("Lot Area Range (sqft)", int(df["LotArea"].min()), int(df["LotArea"].max()), (int(df["LotArea"].min()), int(df["LotArea"].max())))
quality_range = st.sidebar.slider("Overall Quality Range", int(df["OverallQual"].min()), int(df["OverallQual"].max()), (int(df["OverallQual"].min()), int(df["OverallQual"].max())))

# Apply filters
filtered_df = df[(df["YearBuilt"] >= year_range[0]) & (df["YearBuilt"] <= year_range[1])]
filtered_df = filtered_df[(filtered_df["LotArea"] >= lotarea_range[0]) & (filtered_df["LotArea"] <= lotarea_range[1])]
filtered_df = filtered_df[(filtered_df["OverallQual"] >= quality_range[0]) & (filtered_df["OverallQual"] <= quality_range[1])]

# Model on filtered data
filtered_X = filtered_df[numeric_features + categorical_features]
filtered_y = filtered_df[target]
filtered_mse, filtered_r2 = "N/A (Not enough data)", "N/A (Not enough data)"
if len(filtered_df) > 20:  # Increased threshold for stability
    try:
        filtered_X_processed = preprocessor.transform(filtered_X)
        X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(filtered_X_processed, filtered_y, test_size=0.2, random_state=42)
        model_f = Ridge(alpha=1.0)
        model_f.fit(X_train_f, y_train_f)
        y_pred_f = model_f.predict(X_test_f)
        filtered_mse = mean_squared_error(y_test_f, y_pred_f)
        filtered_r2 = r2_score(y_test_f, y_pred_f)
    except Exception as e:
        st.warning(f"Error in filtered model training: {e}")

# Display filtered data
st.subheader("📋 Filtered House Data")
st.write(f"Showing {len(filtered_df)} houses.")
st.dataframe(filtered_df[["LotArea", "YearBuilt", "OverallQual", "SalePrice"]].head(10))

# Model Performance
st.subheader("📊 Model Performance")
st.write(f"**Overall Model:**")
st.write(f"- Mean Squared Error (MSE): {mse:.2f}")
st.write(f"- R² Score: {r2:.4f}")
st.write(f"**Filtered Model:**")
st.write(f"- Mean Squared Error (MSE): {filtered_mse}")
st.write(f"- R² Score: {filtered_r2}")

# Feature Importance (only numeric features)
st.subheader("⚖️ Feature Importance (Coefficients)")
coef_df = pd.DataFrame({"Feature": numeric_features, "Coefficient": model.coef_[:len(numeric_features)]})
st.bar_chart(coef_df.set_index("Feature")["Coefficient"])

# Prediction vs Actual Plot
st.subheader("📉 Actual vs Predicted Prices")
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred, color='blue', alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel("Actual Sale Price")
ax.set_ylabel("Predicted Sale Price")
ax.set_title("Actual vs Predicted House Prices")
st.pyplot(fig)