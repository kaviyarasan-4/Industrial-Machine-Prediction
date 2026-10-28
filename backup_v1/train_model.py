"""
Model Training Script for Industrial Machine Failure Prediction
--------------------------------------------------------------
NOTE: Results demonstrated here are trained on synthetic/simulated
educational sample data, not proprietary real-world factory telemetry.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def train_failure_prediction_model(data_path="machine_data.csv", model_path="model.pkl"):
    print("=" * 65)
    print(" 🤖 TRAINING MACHINE FAILURE PREDICTION MODEL (RANDOM FOREST)")
    print("   (Educational demonstration on synthetic sample data)")
    print("=" * 65)

    # 1. Verify dataset existence
    if not os.path.exists(data_path):
        print(f"\n❌ Error: '{data_path}' does not exist.")
        print("Please run 'python generate_dataset.py' first to create the dataset.")
        return

    # Load dataset
    print(f"\n[Step 1/5] Loading dataset from '{data_path}'...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records.")

    # 2. Define input features (X) and target label (y)
    feature_columns = [
        "air_temperature",
        "process_temperature",
        "rotational_speed",
        "torque",
        "tool_wear"
    ]
    target_column = "machine_failure"

    X = df[feature_columns]
    y = df[target_column]

    print(f"\n[Step 2/5] Input Features (Sensors): {feature_columns}")
    print(f"Target Label: '{target_column}' (0 = Normal, 1 = Failure Risk)")

    # 3. Split into Training (80%) and Testing (20%) sets
    # stratify=y ensures both sets have the exact same proportion of failures
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[Step 3/5] Splitting dataset:")
    print(f" - Training Set: {len(X_train)} samples ({len(X_train)/len(df)*100:.0f}%)")
    print(f" - Testing Set:  {len(X_test)} samples ({len(X_test)/len(df)*100:.0f}%)")

    # 4. Train a Random Forest Classifier
    # class_weight='balanced' helps the model pay extra attention to rare failure cases
    print("\n[Step 4/5] Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    print("✅ Model training complete!")

    # 5. Evaluate the model on the unseen Testing Set
    print("\n[Step 5/5] Evaluating model performance on unseen test data...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "-" * 65)
    print(" 📊 DEMONSTRATION EVALUATION RESULTS (Synthetic Test Data)")
    print("-" * 65)
    print(f"• Accuracy:  {acc * 100:.2f}%  (Overall correct predictions)")
    print(f"• Precision: {prec * 100:.2f}%  (When model predicts failure, how often is it right?)")
    print(f"• Recall:    {rec * 100:.2f}%  (Of all real failures, how many did the model catch?)")
    print(f"• F1-Score:  {f1 * 100:.2f}%  (Harmonic balance between Precision and Recall)")

    print("\nConfusion Matrix:")
    print("                 Predicted Normal (0)   Predicted Failure (1)")
    print(f"Actual Normal  :        {cm[0][0]:<20}    {cm[0][1]}")
    print(f"Actual Failure :        {cm[1][0]:<20}    {cm[1][1]}")

    print("\nFull Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal (0)", "Failure Risk (1)"]))

    # Save the trained model to disk
    joblib.dump(model, model_path)
    print(f"💾 Trained model saved successfully to: '{model_path}'")
    print("=" * 65)

if __name__ == "__main__":
    train_failure_prediction_model()
