
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create output folder
os.makedirs("outputs/week4", exist_ok=True)

# Load dataset
df = pd.read_csv("data/logistics_data.csv", encoding="latin1")

print("=" * 50)
print("WEEK 4: LOGISTICS PREDICTIVE MODELING")
print("=" * 50)

# Select useful features available in the dataset
features = [
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Product Price",
    "Sales",
    "Shipping Mode",
    "Order Region"
]

target = "Days for shipping (real)"

# Keep only required columns
data = df[features + [target]].copy()

# Convert numeric columns
numeric_features = [
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Product Price",
    "Sales"
]

for column in numeric_features + [target]:
    data[column] = pd.to_numeric(data[column], errors="coerce")

# Remove records without the target value
data = data.dropna(subset=[target])

X = data[features]
y = data[target]

categorical_features = ["Shipping Mode", "Order Region"]

# Preprocess numeric and categorical columns
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Models to compare
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
}

results = []
trained_models = {}

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    trained_models[name] = pipeline

    print(f"\n{name}")
    print(f"MAE:  {mae:.3f} days")
    print(f"RMSE: {rmse:.3f} days")
    print(f"R2:   {r2:.3f}")

# Save comparison results
results_df = pd.DataFrame(results).sort_values("MAE")
results_df.to_csv("outputs/week4/model_comparison.csv", index=False)

print("\nMODEL COMPARISON")
print(results_df.round(3).to_string(index=False))

# Cross-validation for the model with the lowest test MAE
best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]

cv_scores = cross_val_score(
    best_model,
    X_train,
    y_train,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=1
)

cv_mae = -cv_scores

print(f"\nBest model by test MAE: {best_name}")
print("5-fold cross-validation MAE scores:",
      np.round(cv_mae, 3))
print("Mean cross-validation MAE:",
      round(cv_mae.mean(), 3), "days")

# Save cross-validation results
pd.DataFrame({
    "Fold": range(1, 6),
    "MAE": cv_mae
}).to_csv("outputs/week4/cross_validation_results.csv", index=False)

# Save a small sample of predictions for review
sample = X_test.head(10).copy()
sample["Actual_Shipping_Days"] = y_test.loc[sample.index]
sample["Predicted_Shipping_Days"] = best_model.predict(
    X_test.loc[sample.index]
)
sample.to_csv("outputs/week4/sample_predictions.csv", index=False)

print("\nFiles saved in outputs/week4/:")
print("- model_comparison.csv")
print("- cross_validation_results.csv")
print("- sample_predictions.csv")
print("\nWeek 4 predictive modeling completed!")
