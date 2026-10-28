# ⚙️ AI-Based Industrial Machine Failure Prediction

A beginner-friendly Predictive Maintenance project built with Python, scikit-learn, and Streamlit. This application demonstrates how machine learning can be used to analyze industrial sensor telemetry and predict potential machine failures before catastrophic breakdowns occur.

> ⚠️ **Important Educational Notice & Disclaimer:**  
> This project uses a **synthetic demonstration dataset** created solely for educational and portfolio purposes. It is **NOT** a certified industrial safety system, SIL-rated controller, or production-grade telemetry engine. It should not be used to operate or safeguard real-life physical machinery.

---

## 📌 Problem Statement
In manufacturing and heavy industry, unexpected machine downtime causes significant financial losses, safety hazards, and production delays. Traditional maintenance approaches are either:
1. **Reactive:** Fixing machines only after they break down (costly and dangerous).
2. **Scheduled / Preventative:** Servicing parts on a fixed schedule regardless of whether they actually need repair (wasteful).

**Predictive Maintenance (PdM)** uses sensor readings—such as temperature, rotational speed, torque, and tool wear—to detect early warning signs of equipment failure, enabling maintenance teams to intervene before breakdowns happen.

---

## 🎯 Objectives
- Build an end-to-end Machine Learning pipeline using Python.
- Generate a reproducible, realistic synthetic dataset representing industrial sensor telemetry.
- Perform exploratory data analysis to inspect data distributions and class balance.
- Train and evaluate a **Random Forest Classifier** to distinguish between normal operations and failure risks.
- Deploy an interactive **Streamlit dashboard** for real-time risk assessment and decision support.

---

## 🛠️ Technologies Used
- **Python**: Core programming language.
- **pandas**: Tabular data manipulation and CSV handling.
- **scikit-learn**: Machine learning model training, evaluation metrics, and train/test split.
- **joblib**: Model serialization and persistence (`model.pkl`).
- **Streamlit**: Web-based interactive graphical user interface.

---

## 📂 Project Structure & Files

| File | Purpose |
| :--- | :--- |
| [`requirements.txt`](./requirements.txt) | Lists required Python libraries (`pandas`, `scikit-learn`, `streamlit`, `joblib`). |
| [`generate_dataset.py`](./generate_dataset.py) | Generates 1,000 synthetic sensor records saved to `machine_data.csv` using a fixed random seed (`42`). |
| [`machine_data.csv`](./machine_data.csv) | The generated dataset containing 5 sensor features and the `machine_failure` label. |
| [`analyze_data.py`](./analyze_data.py) | Inspects the dataset: preview, dimensions, missing value checks, and summary statistics. |
| [`train_model.py`](./train_model.py) | Trains a Random Forest Classifier, evaluates performance metrics, and exports `model.pkl`. |
| [`model.pkl`](./model.pkl) | The serialized trained machine learning model. |
| [`app.py`](./app.py) | Streamlit web application providing interactive inputs, presets, and diagnostic results. |
| [`README.md`](./README.md) | Complete project documentation and guide. |

---

## ⚙️ Installation & Setup

### 1. Open the Integrated Terminal
In Antigravity IDE / VS Code, press `Ctrl` + `` ` `` or select **Terminal** > **New Terminal**.

### 2. Install Dependencies
Run the following command:
```powershell
python -m pip install -r requirements.txt
```

---

## 🚀 Step-by-Step Workflow

### Step 1: Generate the Dataset
Create the synthetic sensor dataset (`machine_data.csv`):
```powershell
python generate_dataset.py
```
*Creates 1,000 simulated records with 5 input features (air temp, process temp, rotational speed, torque, tool wear) and binary failure labels.*

### Step 2: Analyze the Data
Run the exploratory analysis script:
```powershell
python analyze_data.py
```
*Confirms no missing values, shows min/max/average sensor values, and reports ~93.3% normal vs ~6.7% failure cases.*

### Step 3: Train the Machine Learning Model
Train the Random Forest classifier and save `model.pkl`:
```powershell
python train_model.py
```

### Step 4: Launch the Streamlit Dashboard
Start the interactive web interface:
```powershell
python -m streamlit run app.py
```
Open your browser and navigate to: **`http://localhost:8501`**

---

## 📊 Actual Model Evaluation Results (Verified)

These metrics were recorded by evaluating the trained model on an unseen 20% test partition (200 records) of the synthetic dataset:

| Metric | Score | Plain English Meaning |
| :--- | :---: | :--- |
| **Accuracy** | **98.00%** | 196 out of 200 test cases were correctly predicted. |
| **Precision** | **100.00%** | When the model flagged a failure risk, it was correct 100% of the time (0 false alarms). |
| **Recall** | **69.23%** | The model caught 9 out of 13 actual failures (4 missed failures were simulated random noise). |
| **F1-Score** | **81.82%** | The harmonic balance between precision and recall. |

### Confusion Matrix:
```text
                 Predicted Normal (0)   Predicted Failure (1)
Actual Normal  :        187                     0
Actual Failure :          4                     9
```
- **True Negatives (187):** Normal machine operations correctly identified.
- **False Positives (0):** Zero false alarms raised.
- **True Positives (9):** Equipment failure risks successfully caught in advance.
- **False Negatives (4):** Unpredictable random mechanical glitches.

---

## ⚠️ Limitations & Future Work

1. **Synthetic Telemetry:** Data is generated via mathematical distributions and simplified mechanical thresholds. Real industrial machinery exhibits complex vibrations, harmonics, non-linear thermal dynamics, and environmental fluctuations.
2. **Time-Series Dynamics:** The current model evaluates sensor snapshots independently. Real-world predictive maintenance typically benefits from sequential time-series models (e.g., LSTMs, GRUs, or sliding-window feature engineering) to track gradual tool wear and vibration trends over hours or days.
3. **Multi-Class Failure Modes:** The current model classifies binary failure (`0` vs `1`). Future iterations could classify specific root causes (e.g., Heat Dissipation Failure, Power Failure, Overstrain Failure, or Tool Wear Failure).
4. **Not a Certified Safety System:** This software is designed for learning and demonstration. Industrial environments require hardware interlocks and SIL/CE-certified safety systems.
