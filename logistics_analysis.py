"""
Week 1 Logistics Delivery Performance Analysis
Starter implementation for the DataCo Smart Supply Chain dataset.

1. Download the dataset from:
   https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis
2. Put the CSV in data/logistics_data.csv (or update DATA_PATH).
3. Inspect column names and update COLUMN_MAP if needed.
4. Run: python logistics_analysis.py

Important: DataCo's "Days for shipping (real)" and
"Days for shipment (scheduled)" measure shipping duration against a schedule.
Do not describe them as a verified doorstep delivery timestamp.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = Path("data/logistics_data.csv")

# Common column names in the DataCo CSV; verify exact spelling in your copy.
COLUMN_MAP = {
    "actual_days": "Days for shipping (real)",
    "scheduled_days": "Days for shipment (scheduled)",
    "shipping_mode": "Shipping Mode",
    "late_risk": "Late_delivery_risk",
}


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download a permitted CSV dataset "
            "and place it here, or update DATA_PATH."
        )
    df = pd.read_csv(path, encoding="latin1", low_memory=False)
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")
    print("\nColumns:", list(df.columns))
    print("\nMissing values by column:\n",
          df.isna().sum().sort_values(ascending=False).head(20))
    print("\nDuplicate rows:", int(df.duplicated().sum()))
    return df


def calculate_shipping_kpis(df: pd.DataFrame) -> dict:
    actual_col = COLUMN_MAP["actual_days"]
    scheduled_col = COLUMN_MAP["scheduled_days"]

    if actual_col not in df.columns or scheduled_col not in df.columns:
        print(
            "\nShipping KPIs not calculated because required columns were not found. "
            "Inspect the column list and update COLUMN_MAP."
        )
        return {}

    data = df.copy()
    data[actual_col] = pd.to_numeric(data[actual_col], errors="coerce")
    data[scheduled_col] = pd.to_numeric(data[scheduled_col], errors="coerce")
    data = data.dropna(subset=[actual_col, scheduled_col])
    data = data[(data[actual_col] >= 0) & (data[scheduled_col] >= 0)]

    if data.empty:
        print("No valid rows available for shipping KPI calculations.")
        return {}

    on_schedule = data[actual_col] <= data[scheduled_col]
    results = {
        "valid_records": int(len(data)),
        "on_schedule_shipping_rate_pct": round(float(on_schedule.mean() * 100), 2),
        "late_shipment_rate_pct": round(float((~on_schedule).mean() * 100), 2),
        "average_actual_shipping_days": round(float(data[actual_col].mean()), 2),
        "average_scheduled_shipping_days": round(float(data[scheduled_col].mean()), 2),
    }

    print("\nShipping KPIs")
    for key, value in results.items():
        print(f"{key}: {value}")
    return results


def compare_shipping_modes(df: pd.DataFrame) -> None:
    actual_col = COLUMN_MAP["actual_days"]
    scheduled_col = COLUMN_MAP["scheduled_days"]
    mode_col = COLUMN_MAP["shipping_mode"]

    if not all(col in df.columns for col in [actual_col, scheduled_col, mode_col]):
        print("Skipping chart: actual, scheduled, or shipping-mode column is missing.")
        return

    data = df.copy()
    data[actual_col] = pd.to_numeric(data[actual_col], errors="coerce")
    data[scheduled_col] = pd.to_numeric(data[scheduled_col], errors="coerce")
    data = data.dropna(subset=[actual_col, scheduled_col, mode_col])
    if data.empty:
        print("Skipping chart: no valid records.")
        return

    data["is_late_vs_schedule"] = data[actual_col] > data[scheduled_col]
    rates = data.groupby(mode_col)["is_late_vs_schedule"].mean().mul(100).sort_values()
    print("\nLate-shipment rate by shipping mode (%):\n", rates.round(2))
    rates.plot(kind="bar", ylabel="Late-shipment rate (%)",
               title="Late-shipment rate by shipping mode")
    plt.tight_layout()
    plt.show()


def check_late_risk_label(df: pd.DataFrame) -> None:
    actual_col = COLUMN_MAP["actual_days"]
    scheduled_col = COLUMN_MAP["scheduled_days"]
    risk_col = COLUMN_MAP["late_risk"]

    if not all(col in df.columns for col in [actual_col, scheduled_col, risk_col]):
        print("Skipping label check: one or more required columns are missing.")
        return

    data = df.copy()
    data[actual_col] = pd.to_numeric(data[actual_col], errors="coerce")
    data[scheduled_col] = pd.to_numeric(data[scheduled_col], errors="coerce")
    data[risk_col] = pd.to_numeric(data[risk_col], errors="coerce")
    data = data.dropna(subset=[actual_col, scheduled_col, risk_col])
    if data.empty:
        print("Skipping label check: no valid records.")
        return

    derived = (data[actual_col] > data[scheduled_col]).astype(int)
    mismatch_count = int((derived != data[risk_col]).sum())
    print(f"Rows checked against derived late label: {len(data):,}")
    print(f"Rows where derived label differs from Late_delivery_risk: {mismatch_count:,}")
    print("Investigate any mismatches using the dataset's data dictionary.")


def main() -> None:
    df = load_data(DATA_PATH)
    calculate_shipping_kpis(df)
    compare_shipping_modes(df)
    check_late_risk_label(df)


if __name__ == "__main__":
    main()
