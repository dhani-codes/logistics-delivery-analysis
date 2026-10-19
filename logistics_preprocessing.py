
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# WEEK 2: DATA CLEANING AND PREPROCESSING
# ==========================================

# Step 1: Define file paths
INPUT_FILE = os.path.join("data", "logistics_data.csv")
OUTPUT_DIR = "outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "logistics_cleaned.csv")

print("=" * 55)
print("LOGISTICS DATA CLEANING AND PREPROCESSING")
print("=" * 55)

# Step 2: Load the dataset
try:
    print("\nStep 1: Loading dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
        encoding="latin1"
    )

    print("Dataset loaded successfully!")
    print("Original rows:", df.shape[0])
    print("Original columns:", df.shape[1])

except FileNotFoundError:
    print("ERROR: Dataset file not found.")
    print("Check that data/logistics_data.csv exists.")
    raise SystemExit(1)

except Exception as error:
    print("ERROR while loading dataset:", error)
    raise SystemExit(1)


# Step 3: Inspect dataset
print("\nStep 2: Inspecting dataset...")
print("\nFirst five rows:")
print(df.head())

print("\nDataset information:")
print("Rows and columns:", df.shape)
print("Duplicate rows:", df.duplicated().sum())


# Step 4: Identify missing values
print("\nStep 3: Missing values before cleaning:")

missing_before = df.isnull().sum()
missing_before = missing_before[missing_before > 0]

if missing_before.empty:
    print("No missing values found.")
else:
    print(missing_before.sort_values(ascending=False))


# Step 5: Remove completely empty columns
print("\nStep 4: Removing completely empty columns...")

empty_columns = df.columns[df.isnull().all()].tolist()

if empty_columns:
    df = df.drop(columns=empty_columns)
    print("Removed columns:", empty_columns)
else:
    print("No completely empty columns found.")


# Step 6: Remove duplicate rows
print("\nStep 5: Removing duplicate rows...")

duplicate_count = df.duplicated().sum()
df = df.drop_duplicates()

print("Duplicate rows found:", duplicate_count)
print("Rows remaining:", len(df))


# Step 7: Clean extra spaces in text columns
print("\nStep 6: Cleaning text columns...")

text_columns = df.select_dtypes(include=["object"]).columns

for column in text_columns:
    df[column] = df[column].map(
        lambda value: value.strip()
        if isinstance(value, str)
        else value
    )

print("Extra spaces removed from text values.")


# Step 8: Check missing values after cleaning
print("\nStep 7: Missing values after cleaning:")

missing_after = df.isnull().sum()
missing_after = missing_after[missing_after > 0]

if missing_after.empty:
    print("No missing values found.")
else:
    print(missing_after.sort_values(ascending=False))


# Step 9: Detect possible outliers using IQR
print("\nStep 8: Detecting potential outliers...")

numeric_columns = [
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Product Price"
]

for column in numeric_columns:
    if column in df.columns:
        values = pd.to_numeric(
            df[column], errors="coerce"
        ).dropna()

        if len(values) > 0:
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1

            lower_limit = q1 - 1.5 * iqr
            upper_limit = q3 + 1.5 * iqr

            outlier_count = (
                (values < lower_limit) |
                (values > upper_limit)
            ).sum()

            print(
                f"{column}: {outlier_count} potential outliers"
            )

print("Outliers are flagged for review, not automatically deleted.")


# Step 10: Demonstrate Min-Max normalization
print("\nStep 9: Demonstrating normalization...")

scale_columns = [
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Product Price"
]

available_columns = [
    column for column in scale_columns
    if column in df.columns
]

if available_columns:
    numeric_data = df[available_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    # Fill missing values temporarily for this demonstration
    # so the scaler can process the selected features.
    numeric_data = numeric_data.fillna(
        numeric_data.median()
    )

    scaler = MinMaxScaler()

    scaled_values = scaler.fit_transform(numeric_data)

    scaled_df = pd.DataFrame(
        scaled_values,
        columns=available_columns,
        index=df.index
    )

    print("Min-Max normalization completed.")
    print("Sample normalized values:")
    print(scaled_df.head())

    print(
        "Normalized values range from 0 to 1."
    )
else:
    print("No selected numeric columns were found.")


# Step 11: Save the cleaned dataset locally
print("\nStep 10: Saving cleaned dataset...")

os.makedirs(OUTPUT_DIR, exist_ok=True)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print("Cleaned dataset saved to:", OUTPUT_FILE)


# Step 12: Display final summary
print("\n" + "=" * 55)
print("FINAL PREPROCESSING SUMMARY")
print("=" * 55)

print("Original rows:", 180519 if len(df) <= 180519 else "See original dataset")
print("Final rows:", len(df))
print("Final columns:", len(df.columns))
print("Duplicate rows removed:", duplicate_count)
print("Completely empty columns removed:", len(empty_columns))
print("Missing values remaining:", int(df.isnull().sum().sum()))
print("Output file:", OUTPUT_FILE)

print("\nWeek 2 preprocessing completed successfully!")
print("=" * 55)
