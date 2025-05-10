import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Load processed data
try:
    X = np.load("X_processed.npy")
    y = np.load("y.npy")
except FileNotFoundError:
    print("Error: Processed data files not found. Run preprocess.py first.")
    exit()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.4f}")

# Save model (optional)
# import joblib
# joblib.dump(model, "ridge_model.pkl")