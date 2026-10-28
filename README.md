# ⚙️ AI-Based Industrial Fault Diagnosis and Root Cause Analysis System

A comprehensive industrial predictive maintenance and telemetry analysis system built using **Python**, **scikit-learn**, and **Streamlit**.

This application transforms standard binary failure prediction into a multi-class **Fault Diagnosis and Root Cause Analysis (RCA)** engine. It analyzes five real-time machine telemetry channels to classify specific industrial failure modes, flags sensor anomalies against nominal baselines, identifies potential physical contributing factors, and generates prescriptive maintenance checklists.

> ⚠️ **Important Educational Notice & Disclaimer:**  
> This project uses an explicitly **synthetic demonstration dataset** created solely for learning and portfolio demonstration. The sensor values and fault patterns reflect realistic physical assumptions but do **NOT** represent proprietary factory data. This application is **NOT a certified industrial safety system** (e.g., SIL/CE rated) and must never be used to control or safeguard real industrial machinery.

---

## 📌 Problem Statement & Overview
Modern manufacturing operations rely on continuous machinery where unpredicted downtime leads to massive financial loss and operator safety hazards. Traditional monitoring relies on:
1. **Generic Alarms:** Alerting that "something is wrong" without specifying what or why.
2. **Post-Mortem Investigation:** Requiring hours of manual troubleshooting by senior technicians to diagnose the root cause after the machine has already halted.

### The Solution: Fault Diagnosis & Root Cause Analysis
This system implements:
* **Multi-Class Machine Learning:** Instead of a generic `Normal` vs `Failure` flag, the model predicts the exact fault mode.
* **Sensor Anomaly Detection:** Compares every individual sensor against historical nominal boundaries.
* **Root Cause Analysis (RCA):** Explains *why* the sensors triggered the alert by identifying potential physical mechanisms.
* **Prescriptive Maintenance Action Checklist:** Recommends targeted, step-by-step procedures to the maintenance team.

---

## 🎯 Fault Categories Classified
The model classifies five distinct operational states:

| Category | Sensor Signature / Physics Cause |
| :--- | :--- |
| 🟢 **Normal Operation** | All sensor telemetry within nominal tolerances. |
| 🔥 **Overheating** | Elevated process temperature or high process-to-ambient thermal delta ($\Delta T$). |
| ⚡ **Excessive Torque** | High mechanical drive resistance / overstrain ($> 60 \text{ Nm}$). |
| ⏱️ **Excessive Tool Wear** | Cutter service duration exceeding recommended life ($> 200 \text{ min}$). |
| 🔄 **Abnormal Rotational Speed** | Significant spindle deviation from nominal speed ($< 1200$ or $> 2200 \text{ RPM}$). |

---

## 🛠️ Architecture & Technologies

```
                  ┌───────────────────────────────────────────────┐
                  │ 5 Machine Telemetry Inputs (Sensors)          │
                  │ [Air Temp, Process Temp, RPM, Torque, Wear]   │
                  └───────────────────────┬───────────────────────┘
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
   ┌───────────────────────────────┐             ┌───────────────────────────────┐
   │ Machine Learning Layer        │             │ Diagnostic & Rule Layer       │
   │ (Random Forest Classifier)    │             │ (Statistical Baseline Engine) │
   └──────────────┬────────────────┘             └───────────────┬───────────────┘
                  │                                              │
                  ▼                                              ▼
   [Multi-Class Fault Diagnosis]                 [Sensor Anomaly Table]
   [Confidence Probabilities]                    [Possible Physical Causes (RCA)]
                                                 [Actionable Maintenance Steps]
```

- **Python**: Core implementation.
- **pandas**: Dataset processing and feature extraction.
- **scikit-learn**: Random Forest classifier, multi-class evaluation metrics, and train/test splits.
- **joblib**: Model packaging and metadata bundling (`model.pkl`).
- **Streamlit**: Web dashboard with interactive presets, anomaly indicators, and diagnostic panels.

---

## 📂 Project Structure

```text
Industrial-Machine-Prediction/
├── backup_v1/               # Preserved backup of original binary prediction files
├── requirements.txt         # Project dependencies
├── machine_data.csv         # 1,200 synthetic telemetry records with 5 fault categories
├── generate_dataset.py      # Reproducible synthetic dataset generator (Seed: 42)
├── analyze_data.py          # Exploratory data analysis & category distribution report
├── train_model.py           # Multi-class Random Forest training & evaluation script
├── model.pkl                # Trained model bundle (model, features, classes, baseline stats)
├── app.py                   # Streamlit fault diagnosis & RCA web application
└── README.md                # Comprehensive documentation
```

---

## ⚙️ Installation & Running the Project

### 1. Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### 2. Generate the Demonstration Dataset
```powershell
python generate_dataset.py
```
*Generates 1,200 synthetic records with balanced multi-class fault distributions (80% Normal, 5% each fault category).*

### 3. Run Exploratory Analysis
```powershell
python analyze_data.py
```

### 4. Train the Multi-Class Model
```powershell
python train_model.py
```

### 5. Launch the Streamlit Dashboard
```powershell
python -m streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 📊 Actual Verified Model Evaluation Results

*(Evaluated on an unseen 20% stratified test partition of 240 records from the demonstration dataset)*

* **Overall Multi-Class Accuracy:** **97.92%**
* **Macro-Averaged Precision:** **94.92%**
* **Macro-Averaged Recall:** **94.79%**
* **Macro-Averaged F1-Score:** **94.79%**

### Full Classification Report:
```text
                           precision    recall  f1-score   support

Abnormal Rotational Speed     1.0000    0.9167    0.9565        12
      Excessive Tool Wear     1.0000    1.0000    1.0000        12
         Excessive Torque     0.9231    1.0000    0.9600        12
         Normal Operation     0.9896    0.9896    0.9896       192
              Overheating     0.8333    0.8333    0.8333        12

                 accuracy                         0.9792       240
                macro avg     0.9492    0.9479    0.9479       240
             weighted avg     0.9795    0.9792    0.9792       240
```

---

## ⚠️ Limitations & Technical Assumptions

1. **Synthetic Demonstration Data:** The dataset is generated via mathematical distributions and simplified physical rules. Real industrial environments exhibit non-linear acoustic harmonics, vibration frequencies, and complex ambient temperature fluctuations.
2. **Distinction Between Prediction and Causation:** The machine learning model identifies *statistical correlation* in sensor patterns. The root-cause explanations and maintenance suggestions are *rule-based engineering hypotheses* to guide human technicians, not certified physical proofs.
3. **Non-Certified Demonstration:** This application does not interface with physical safety interlocks or programmable logic controllers (PLCs) and must not be used as a primary safety mechanism.
