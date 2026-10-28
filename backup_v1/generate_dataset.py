"""
Dataset Generator for Industrial Machine Failure Prediction Project
-------------------------------------------------------------------
NOTE: This script generates a synthetic (simulated) sample dataset for educational
and demonstration purposes. It does NOT claim to represent actual real-world
proprietary industrial telemetry data.
"""

import numpy as np
import pandas as pd

def generate_machine_dataset(num_records=1000, random_seed=42):
    # Set a fixed random seed so the generated numbers are identical every time you run it
    np.random.seed(random_seed)
    
    print(f"Generating {num_records} simulated machine sensor records...")
    
    # 1. Air Temperature in Kelvin (Typical ambient factory temp: ~295K to 304K / ~22°C to 31°C)
    air_temperature = np.round(np.random.normal(loc=300.0, scale=2.0, size=num_records), 1)
    
    # 2. Process Temperature in Kelvin (Typically ~10K higher than air temperature)
    temp_difference = np.random.normal(loc=10.0, scale=1.0, size=num_records)
    process_temperature = np.round(air_temperature + temp_difference, 1)
    
    # 3. Rotational Speed in RPM (Rotations Per Minute, typical range: 1200 - 2800 RPM)
    rotational_speed = np.random.normal(loc=1530.0, scale=180.0, size=num_records).astype(int)
    rotational_speed = np.clip(rotational_speed, 1150, 2900)
    
    # 4. Torque in Newton-meters (Nm) (Twisting force, typical range: 15 - 75 Nm)
    # Physics relationship: Higher torque usually corresponds to lower rotational speed
    torque = np.round(np.random.normal(loc=40.0, scale=10.0, size=num_records), 1)
    torque = np.clip(torque, 10.0, 80.0)
    
    # 5. Tool Wear in minutes (How long the cutting/working tool has been used, 0 to 250 minutes)
    tool_wear = np.random.randint(0, 240, size=num_records)
    
    # 6. Failure Logic (machine_failure: 0 = Normal, 1 = Failure Risk)
    # Realistic failure causes:
    # - Tool wear failure: Tool used too long (> 200 mins) under high torque (> 50 Nm)
    # - Heat dissipation failure: Air and process temperatures are too close or too hot
    # - Power / Overstrain failure: Extreme torque (> 65 Nm) or extreme rotational speed
    # - Small random baseline failure chance (e.g. random electrical glitch)
    
    failure_risk = np.zeros(num_records, dtype=int)
    
    for i in range(num_records):
        # Condition A: Tool wear fatigue under high torque
        cond_tool_wear = (tool_wear[i] > 190) and (torque[i] > 50.0)
        
        # Condition B: Overheat / heat dissipation strain
        cond_heat = (process_temperature[i] - air_temperature[i] < 8.0) and (rotational_speed[i] < 1350)
        
        # Condition C: Mechanical overstrain (extreme torque)
        cond_overstrain = torque[i] > 65.0
        
        # Condition D: Random baseline failure chance (~2%)
        cond_random = np.random.rand() < 0.02
        
        if cond_tool_wear or cond_heat or cond_overstrain or cond_random:
            failure_risk[i] = 1
        else:
            failure_risk[i] = 0
            
    # Combine everything into a pandas DataFrame (a structured table)
    df = pd.DataFrame({
        "air_temperature": air_temperature,
        "process_temperature": process_temperature,
        "rotational_speed": rotational_speed,
        "torque": torque,
        "tool_wear": tool_wear,
        "machine_failure": failure_risk
    })
    
    return df

if __name__ == "__main__":
    # Generate the 1,000 records
    dataset = generate_machine_dataset(num_records=1000, random_seed=42)
    
    # Save the dataset to a CSV file in the current directory
    output_filename = "machine_data.csv"
    dataset.to_csv(output_filename, index=False)
    
    # Print a summary of the generated dataset
    total_records = len(dataset)
    normal_count = (dataset["machine_failure"] == 0).sum()
    failure_count = (dataset["machine_failure"] == 1).sum()
    
    print("\n✅ Dataset successfully generated and saved to:", output_filename)
    print(f"Total Records: {total_records}")
    print(f"Normal Records (0): {normal_count} ({(normal_count/total_records)*100:.1f}%)")
    print(f"Failure Records (1): {failure_count} ({(failure_count/total_records)*100:.1f}%)")
    print("\nFirst 5 rows preview:")
    print(dataset.head())
