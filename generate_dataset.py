"""
Dataset Generator for Industrial Fault Diagnosis & Root Cause Analysis System
----------------------------------------------------------------------------
NOTE & DISCLAIMER:
This script generates a synthetic demonstration dataset for educational and
portfolio purposes. It simulates realistic sensor relationships (temperatures,
rotational speed, torque, tool wear) but does NOT represent real-world factory
telemetry and is NOT a certified industrial safety system.
"""

import numpy as np
import pandas as pd

def generate_fault_diagnosis_dataset(num_records=1200, random_seed=42):
    """
    Generates synthetic industrial telemetry data with explicit multi-class fault categories:
      1. Normal Operation
      2. Overheating
      3. Excessive Torque
      4. Excessive Tool Wear
      5. Abnormal Rotational Speed
    """
    np.random.seed(random_seed)
    print(f"Generating {num_records} synthetic industrial telemetry records...")

    # 1. Base Sensor Readings (Normal Distributions)
    # Air Temperature in Kelvin (~296K to 304K / ~23°C to 31°C)
    air_temp = np.random.normal(loc=300.0, scale=2.0, size=num_records)
    
    # Process Temperature in Kelvin (Normally ~10K above ambient air temperature)
    temp_diff = np.random.normal(loc=10.0, scale=1.0, size=num_records)
    proc_temp = air_temp + temp_diff
    
    # Rotational Speed in RPM (Nominal baseline: ~1530 RPM)
    rot_speed = np.random.normal(loc=1530.0, scale=120.0, size=num_records).astype(int)
    
    # Torque in Nm (Nominal baseline: ~40 Nm)
    torque = np.random.normal(loc=40.0, scale=8.0, size=num_records)
    
    # Tool Wear in minutes (0 to 180 min under normal duty cycles)
    tool_wear = np.random.randint(0, 180, size=num_records)

    # 2. Inject Controlled Multi-Class Fault Patterns
    # We assign distinct fault categories so the ML classifier learns real diagnostic patterns
    fault_categories = ["Normal Operation"] * num_records
    
    # Define target allocations for faults (approx. 5% each, ~60 samples per fault out of 1200)
    indices = np.random.permutation(num_records)
    
    idx_overheating = indices[0:60]
    idx_torque = indices[60:120]
    idx_wear = indices[120:180]
    idx_speed = indices[180:240]
    # Remaining indices (240:1200 = 960 records, 80%) are "Normal Operation"

    # Category 1: Overheating
    # Physics Assumption: Process temperature rises abnormally high relative to air temp, or air temp is excessive
    for i in idx_overheating:
        proc_temp[i] += np.random.uniform(7.0, 14.0)   # Excess process heat
        air_temp[i] += np.random.uniform(2.0, 5.0)     # High ambient factory heat
        fault_categories[i] = "Overheating"

    # Category 2: Excessive Torque
    # Physics Assumption: Heavy mechanical bind, motor overstrain, or abnormal resistance
    for i in idx_torque:
        torque[i] = np.random.uniform(62.0, 82.0)      # High mechanical strain
        rot_speed[i] = int(np.random.uniform(1180, 1380)) # Motor slows under extreme drag
        fault_categories[i] = "Excessive Torque"

    # Category 3: Excessive Tool Wear
    # Physics Assumption: Cutter or bit has exceeded maximum service life without tool change
    for i in idx_wear:
        tool_wear[i] = np.random.randint(205, 255)     # Severely worn tool bit
        torque[i] += np.random.uniform(5.0, 12.0)      # Worn cutter creates extra friction
        fault_categories[i] = "Excessive Tool Wear"

    # Category 4: Abnormal Rotational Speed
    # Physics Assumption: Spindle bearing instability, drive slip, or governor failure
    for i in idx_speed:
        if np.random.rand() > 0.5:
            rot_speed[i] = int(np.random.uniform(2350, 2850)) # Spindle runaway / overspeed
        else:
            rot_speed[i] = int(np.random.uniform(900, 1150))  # Drive stall / severe underspeed
        fault_categories[i] = "Abnormal Rotational Speed"

    # Round sensor telemetry to realistic precision
    air_temp = np.round(air_temp, 1)
    proc_temp = np.round(proc_temp, 1)
    torque = np.round(torque, 1)
    rot_speed = np.clip(rot_speed, 850, 3000)
    torque = np.clip(torque, 8.0, 90.0)

    # 3. Backward-compatible binary failure indicator (0 = Normal, 1 = Fault)
    failure_flag = [0 if f == "Normal Operation" else 1 for f in fault_categories]

    # Assemble into DataFrame
    df = pd.DataFrame({
        "air_temperature": air_temp,
        "process_temperature": proc_temp,
        "rotational_speed": rot_speed,
        "torque": torque,
        "tool_wear": tool_wear,
        "fault_category": fault_categories,
        "machine_failure": failure_flag
    })

    return df

if __name__ == "__main__":
    df = generate_fault_diagnosis_dataset(num_records=1200, random_seed=42)
    output_path = "machine_data.csv"
    df.to_csv(output_path, index=False)

    print(f"\n✅ Dataset successfully generated and saved to: {output_path}")
    print(f"Total Records: {len(df)}")
    print("\nFault Category Distribution:")
    print(df["fault_category"].value_counts())
    print(f"\nNormal vs Failure ratio:")
    print(df["machine_failure"].value_counts(normalize=True).round(3) * 100)
    print("\nPreview:")
    print(df.head())
