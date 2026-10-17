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
