
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create an output folder for visualization images
os.makedirs("outputs/week3", exist_ok=True)

# Load the logistics dataset
file_path = "data/logistics_data.csv"
df = pd.read_csv(file_path, encoding="latin1")

print("=" * 50)
print("WEEK 3: LOGISTICS DATA ANALYSIS")
print("=" * 50)

# Select useful logistics variables
columns = [
    "Days for shipment (scheduled)",
    "Days for shipping (real)",
    "Order Item Quantity",
    "Product Price",
    "Sales"
]

data = df[columns].copy()

# Give columns easier names
data.columns = [
    "Scheduled_Days",
    "Actual_Shipping_Days",
    "Quantity",
    "Product_Price",
    "Sales"
]

# Convert selected columns to numeric values
for column in data.columns:
    data[column] = pd.to_numeric(data[column], errors="coerce")

# Remove rows missing values in the selected analysis variables
data = data.dropna()

print("\nDATASET OVERVIEW")
print("Rows analyzed:", len(data))
print("\nFirst five rows:")
print(data.head())

print("\nDESCRIPTIVE STATISTICS")
print(data.describe().round(2))

print("\nMEDIAN VALUES")
print(data.median().round(2))

# Calculate shipping delay
data["Shipping_Delay"] = (
    data["Actual_Shipping_Days"] - data["Scheduled_Days"]
)

print("\nSHIPPING PERFORMANCE")
print("Average scheduled shipping days:",
      round(data["Scheduled_Days"].mean(), 2))
print("Average actual shipping days:",
      round(data["Actual_Shipping_Days"].mean(), 2))
print("Average shipping delay:",
      round(data["Shipping_Delay"].mean(), 2))
print("Late shipment percentage:",
      round((data["Shipping_Delay"] > 0).mean() * 100, 2), "%")

# Set a consistent visual style
sns.set_theme(style="whitegrid")

# Visualization 1: Distribution of actual shipping days
plt.figure(figsize=(8, 5))
sns.histplot(data=data, x="Actual_Shipping_Days", bins=15, kde=True)
plt.title("Distribution of Actual Shipping Days")
plt.xlabel("Actual Shipping Days")
plt.ylabel("Number of Shipments")
plt.tight_layout()
plt.savefig("outputs/week3/shipping_distribution.png", dpi=300)
plt.close()

# Visualization 2: Scheduled versus actual shipping days
shipping_comparison = data[
    ["Scheduled_Days", "Actual_Shipping_Days"]
].mean().reset_index()
shipping_comparison.columns = ["Shipping_Type", "Average_Days"]

plt.figure(figsize=(8, 5))
sns.barplot(
    data=shipping_comparison,
    x="Shipping_Type",
    y="Average_Days"
)
plt.title("Average Scheduled vs Actual Shipping Days")
plt.xlabel("Shipping Measure")
plt.ylabel("Average Days")
plt.tight_layout()
plt.savefig("outputs/week3/shipping_comparison.png", dpi=300)
plt.close()

# Visualization 3: Shipping delay distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=data, x="Shipping_Delay", bins=20, kde=True)
plt.axvline(0, linestyle="--", color="red", label="On schedule")
plt.title("Distribution of Shipping Delays")
plt.xlabel("Actual Days - Scheduled Days")
plt.ylabel("Number of Shipments")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/week3/shipping_delay.png", dpi=300)
plt.close()

# Visualization 4: Correlation heatmap
correlation = data[
    ["Scheduled_Days", "Actual_Shipping_Days",
     "Quantity", "Product_Price", "Sales"]
].corr()

print("\nCORRELATION MATRIX")
print(correlation.round(2))

plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm",
            fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Between Logistics Variables")
plt.tight_layout()
plt.savefig("outputs/week3/correlation_heatmap.png", dpi=300)
plt.close()

# Visualization 5: Product price distribution
plt.figure(figsize=(8, 5))
sns.boxplot(data=data, x="Product_Price")
plt.title("Distribution and Potential Outliers in Product Price")
plt.xlabel("Product Price")
plt.tight_layout()
plt.savefig("outputs/week3/product_price_boxplot.png", dpi=300)
plt.close()

# Save summary statistics for documentation
data.describe().round(2).to_csv(
    "outputs/week3/descriptive_statistics.csv"
)

print("\nVISUALIZATIONS SAVED")
print("1. shipping_distribution.png")
print("2. shipping_comparison.png")
print("3. shipping_delay.png")
print("4. correlation_heatmap.png")
print("5. product_price_boxplot.png")
print("\nAll output files are in outputs/week3/")
print("Week 3 analysis completed successfully!")
