"""
Multi-Class Model Training Script for Industrial Fault Diagnosis & Root Cause Analysis
--------------------------------------------------------------------------------------
NOTE: This model is trained on a synthetic demonstration dataset for educational
and portfolio use. It is NOT a certified industrial safety system.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

def train_fault_diagnosis_model(data_path="machine_data.csv", model_path="model.pkl"):
    print("=" * 70)
    print(" 🤖 TRAINING MULTI-CLASS INDUSTRIAL FAULT DIAGNOSIS MODEL")
    print("   (Educational demonstration on synthetic sample data)")
    print("=" * 70)

    if not os.path.exists(data_path):
        print(f"\n❌ Error: '{data_path}' does not exist.")
        print("Please run 'python generate_dataset.py' first.\n")
        return

    # 1. Load dataset
    print(f"\n[Step 1/5] Loading dataset from '{data_path}'...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} total records.")

    # 2. Features and Target
    feature_columns = [
        "air_temperature",
        "process_temperature",
        "rotational_speed",
        "torque",
        "tool_wear"
    ]
    target_column = "fault_category"

    X = df[feature_columns]
    y = df[target_column]

    print(f"\n[Step 2/5] Input Features: {feature_columns}")
    print(f"Target Column: '{target_column}' (5 Categories)")

    # 3. Stratified Train / Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[Step 3/5] Splitting dataset (stratified):")
    print(f" • Training Set: {len(X_train)} samples")
    print(f" • Testing Set:  {len(X_test)} samples")

    # Calculate baseline sensor statistics for Normal Operation from training data
    # Used later by the dashboard for rule-based anomaly detection
    normal_training_df = X_train[y_train == "Normal Operation"]
    normal_stats = {
        col: {
            "mean": float(normal_training_df[col].mean()),
            "std": float(normal_training_df[col].std()),
            "min": float(normal_training_df[col].min()),
            "max": float(normal_training_df[col].max()),
            "p5": float(normal_training_df[col].quantile(0.05)),
            "p95": float(normal_training_df[col].quantile(0.95))
        }
        for col in feature_columns
    }

    # 4. Train Random Forest Classifier
    print("\n[Step 4/5] Training Multi-Class Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=120,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    print("✅ Training complete!")

    # 5. Evaluate on unseen test partition
    print("\n[Step 5/5] Evaluating model performance on unseen test data...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "-" * 70)
    print(" 📊 ACTUAL EVALUATION RESULTS (Unseen Test Partition, N = 240)")
    print("-" * 70)
    print(f"• Overall Multi-Class Accuracy: {accuracy * 100:.2f}%\n")

    labels = sorted(list(y.unique()))
    print("Full Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)

    # Save model bundle
    bundle = {
        "model": model,
        "features": feature_columns,
        "classes": labels,
        "normal_stats": normal_stats
    }
    joblib.dump(bundle, model_path)
    print(f"\n💾 Model bundle saved successfully to: '{model_path}'")
    print("=" * 70)

if __name__ == "__main__":
    train_fault_diagnosis_model()
