import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Load data with error handling
try:
    df = pd.read_csv("D:/machine_learning/house_price_pred/src/train.csv")
except FileNotFoundError:
    print("Error: 'train.csv' not found. Please download it from Kaggle and place it in the project folder.")
    exit()

# Select features and target
numeric_features = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd", "OverallQual", "GrLivArea"]
categorical_features = ["Neighborhood", "SaleCondition"]
target = "SalePrice"

# Check if features exist in the dataset
missing_features = [feat for feat in numeric_features + categorical_features if feat not in df.columns]
if missing_features:
    print(f"Error: These features are missing in the dataset: {missing_features}")
    exit()

X = df[numeric_features + categorical_features]
y = df[target]

# Preprocessing pipeline
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
    np.save("X_processed.npy", X_processed)
    np.save("y.npy", y.values)
    print(f"Preprocessing done. X shape: {X_processed.shape}")
except Exception as e:
    print(f"Error during preprocessing: {e}")
    exit()