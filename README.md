# Logistics Delivery Performance and Delay Analysis

## Project Overview

This project analyzes logistics shipping performance using Python and the DataCo Smart Supply Chain dataset. It explores shipping delays, compares actual shipping duration with scheduled duration, and identifies shipping modes that may need further investigation.

## Objectives

- Measure the on-schedule shipping rate and late-shipment rate.
- Compare actual and scheduled shipping days.
- Analyze late-shipment rates across shipping modes.
- Identify missing values and duplicate records.
- Provide data-driven recommendations for logistics improvement.

## Dataset

The project uses the DataCo Smart Supply Chain dataset, available on Kaggle:

https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis

The dataset contains 180,519 records and 53 columns in the version analyzed. The CSV is stored locally and is not included in this repository.

## Tools and Technologies

- Python 3.10
- pandas
- matplotlib
- scikit-learn

## Key Findings

- Total records analyzed: 180,519
- On-schedule shipping rate: 42.72%
- Late-shipment rate: 57.28%
- Average actual shipping duration: 3.50 days
- Average scheduled shipping duration: 2.93 days
- Duplicate rows: 0

### Late-Shipment Rate by Shipping Mode

- Standard Class: 39.77%
- Same Day: 47.83%
- Second Class: 79.73%
- First Class: 100.00%

These rates are descriptive findings from the analyzed dataset. They do not establish that shipping mode causes delays.

## How to Run

1. Install Python 3.10 or a compatible Python version.
2. Install the dependencies:

   `python -m pip install -r requirements.txt`

3. Place the dataset CSV in the `data` folder and name it `logistics_data.csv`.
4. Run the analysis script:

   `python logistics_analysis.py`

On Windows, use your installed Python executable if the `python` command points to an older version.

## Limitations

This analysis compares actual shipping days with scheduled shipping days. It does not independently verify doorstep delivery dates, and the results do not establish causal relationships.

## Future Improvements

- Create additional visualizations.
- Investigate factors associated with shipping delays.
- Develop and evaluate a machine-learning model for delay-risk prediction.
- Build a dashboard for logistics performance monitoring.

## Author

Student project for a Week 1 logistics strategic planning and data exploration internship task.



## Week 2: Data Collection, Cleaning, and Preprocessing

### Objective
The objective of Week 2 was to understand data quality and perform basic data cleaning and preprocessing on a logistics dataset using Python. The work included identifying missing values, checking duplicate records, removing completely empty columns, detecting potential outliers, and demonstrating numerical normalization.

### Dataset
The project uses the DataCo Smart Supply Chain dataset from Kaggle. The original dataset contains 180,519 rows and 53 columns. The dataset was loaded locally using Python and pandas.

Dataset source: https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis

### Data Cleaning and Preprocessing
A Python script named `logistics_preprocessing.py` was created to automate the preprocessing workflow. The script checked the dataset structure, identified missing values, and checked for duplicate rows. No duplicate rows were found. The completely empty `Product Description` column was removed, reducing the dataset from 53 columns to 52 columns while retaining all 180,519 rows.

Missing values remain in certain columns, including `Order Zipcode`, `Customer Lname`, and `Customer Zipcode`. These values were not automatically filled because missing information should be handled according to its meaning and suitability for analysis.

The script also used the Interquartile Range (IQR) method to flag potential outliers in selected numerical columns. It flagged 2,048 potential outliers in `Product Price`. These were treated as candidates for further investigation rather than automatically removed errors.

Min-max normalization was demonstrated on selected numerical features using scikit-learn. The demonstration used median imputation for missing feature values in the normalization sample only; it did not overwrite the original feature values in the cleaned dataset.

### Results
- Original dataset: 180,519 rows and 53 columns.
- Final working dataset: 180,519 rows and 52 columns.
- Duplicate rows removed: 0.
- Completely empty columns removed: 1.
- Missing values remaining after cleaning: 155,690.
- Potential outliers flagged in `Product Price`: 2,048.
- Cleaned working copy saved locally as `outputs/logistics_cleaned.csv`.

### Conclusion
This task demonstrated that data quality checks are an essential part of logistics analysis. Missing values, duplicates, and potential outliers must be investigated before drawing conclusions. The preprocessing script provides a repeatable starting point for preparing the dataset for further analysis. The original and cleaned CSV files remain local and are excluded from GitHub; only the Python source code and project documentation are shared publicly.
