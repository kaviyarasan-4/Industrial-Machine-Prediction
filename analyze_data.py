"""
Data Analysis Script for Industrial Fault Diagnosis & Root Cause Analysis System
--------------------------------------------------------------------------------
NOTE: This analysis is performed on synthetic/simulated demonstration data
created for educational purposes, not real-world factory telemetry.
"""

import os
import pandas as pd

def analyze_dataset(csv_path="machine_data.csv"):
    print("=" * 65)
    print(" 🏭 INDUSTRIAL FAULT DIAGNOSIS DATASET - EXPLORATORY ANALYSIS")
    print("   (Note: Educational synthetic demonstration data)")
    print("=" * 65)

    if not os.path.exists(csv_path):
        print(f"\n❌ Error: '{csv_path}' was not found in the current directory.")
        print("Please run 'python generate_dataset.py' first.\n")
        return

    df = pd.read_csv(csv_path)

    # 1. Preview
    print("\n📌 1. FIRST 5 ROWS:")
    print("-" * 65)
    print(df.head())

    # 2. Dimensions
    rows, cols = df.shape
    print("\n📌 2. DATASET DIMENSIONS:")
    print("-" * 65)
    print(f"Total Records: {rows}")
    print(f"Total Columns: {cols}")
    print(f"Columns: {list(df.columns)}")

    # 3. Missing values
    print("\n📌 3. MISSING VALUES CHECK:")
    print("-" * 65)
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✅ Zero missing values found across all columns.")
    else:
        print(missing)

    # 4. Multi-Class Fault Breakdown
    print("\n📌 4. FAULT CATEGORY DISTRIBUTION (MULTI-CLASS TARGET):")
    print("-" * 65)
    category_counts = df["fault_category"].value_counts()
    for cat, count in category_counts.items():
        pct = (count / rows) * 100
        print(f"• {cat:<28} : {count:4d} samples ({pct:5.1f}%)")

    # 5. Sensor statistics grouped by fault category
    print("\n📌 5. MEAN SENSOR VALUES BY FAULT CATEGORY:")
    print("-" * 65)
    grouped_means = df.groupby("fault_category")[[
        "air_temperature",
        "process_temperature",
        "rotational_speed",
        "torque",
        "tool_wear"
    ]].mean().round(1)
    print(grouped_means)

    print("\n" + "=" * 65)
    print("✅ Analysis complete! Dataset is ready for multi-class model training.")
    print("=" * 65)

if __name__ == "__main__":
    analyze_dataset("machine_data.csv")
