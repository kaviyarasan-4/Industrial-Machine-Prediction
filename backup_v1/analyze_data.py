"""
Data Analysis Script for Industrial Machine Failure Prediction
--------------------------------------------------------------
NOTE: This analysis is performed on synthetic/simulated sample data
created for educational purposes, not real-world factory telemetry.
"""

import pandas as pd
import os

def analyze_dataset(csv_path="machine_data.csv"):
    print("=" * 60)
    print(" 🏭 INDUSTRIAL MACHINE DATASET - EXPLORATORY ANALYSIS")
    print("   (Note: Educational synthetic sample data)")
    print("=" * 60)

    # 1. Check if the CSV file exists
    if not os.path.exists(csv_path):
        print(f"\n❌ Error: '{csv_path}' was not found in the current directory.")
        print("Please run 'python generate_dataset.py' first to create the dataset.\n")
        return

    # Load the CSV file into a pandas DataFrame (read-only, does not modify CSV)
    df = pd.read_csv(csv_path)

    # 2. Display the first 5 rows
    print("\n📌 1. FIRST 5 ROWS OF DATA:")
    print("-" * 60)
    print(df.head())

    # 3. Display the number of rows and columns
    rows, cols = df.shape
    print("\n📌 2. DATASET DIMENSIONS:")
    print("-" * 60)
    print(f"Total Rows (Records):    {rows}")
    print(f"Total Columns (Features): {cols}")
    print(f"Column Names: {list(df.columns)}")

    # 4. Check for missing values
    print("\n📌 3. MISSING VALUES CHECK:")
    print("-" * 60)
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    if total_missing == 0:
        print("✅ Excellent! No missing values found in any column.")
    else:
        print(missing_counts)

    # 5 & 6. Count normal records and failure records
    print("\n📌 4. MACHINE STATUS BREAKDOWN (TARGET VARIABLE):")
    print("-" * 60)
    status_counts = df["machine_failure"].value_counts()
    normal_count = status_counts.get(0, 0)
    failure_count = status_counts.get(1, 0)
    
    normal_pct = (normal_count / rows) * 100
    failure_pct = (failure_count / rows) * 100

    print(f"🟢 Normal Machines  (machine_failure = 0): {normal_count:4d} ({normal_pct:.1f}%)")
    print(f"🔴 Failure Machines (machine_failure = 1): {failure_count:4d} ({failure_pct:.1f}%)")
    print("\n💡 Beginner Note: In industrial manufacturing, machine failures are rare.")
    print("   Having around 5-10% failure cases is very typical and realistic.")

    # 7. Basic statistics for all numeric columns
    print("\n📌 5. SUMMARY STATISTICS (Min, Max, Average):")
    print("-" * 60)
    stats = df.describe().round(2)
    print(stats)

    print("\n" + "=" * 60)
    print("✅ Analysis complete! The dataset is clean and ready for ML training.")
    print("=" * 60)

if __name__ == "__main__":
    analyze_dataset("machine_data.csv")
