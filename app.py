"""
Streamlit Web Dashboard for Industrial Fault Diagnosis & Root Cause Analysis System
-----------------------------------------------------------------------------------
NOTE: This web application is an educational demonstration built on synthetic
telemetry data. It is NOT a certified industrial safety system and should NOT
be used to control real-life industrial machinery.
"""

import os
import joblib
import pandas as pd
import streamlit as st

# Configure page layout and title
st.set_page_config(
    page_title="Industrial Fault Diagnosis & RCA System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for modern, clean interface
st.markdown("""
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .badge-normal {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-warning {
        background-color: #FEF9C3;
        color: #854D0E;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-critical {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .card-normal {
        background-color: #F0FDF4;
        border: 2px solid #22C55E;
        border-radius: 10px;
        padding: 1.4rem;
        color: #14532D;
    }
    .card-fault {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 10px;
        padding: 1.4rem;
        color: #7F1D1D;
    }
    .rca-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
        color: #1E293B;
    }
    .maintenance-box {
        background-color: #FDF4FF;
        border-left: 4px solid #A855F7;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
        color: #581C87;
    }
    .disclaimer-box {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        font-size: 0.9rem;
        color: #92400E;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

MODEL_FILE = "model.pkl"

@st.cache_resource
def load_bundle():
    if not os.path.exists(MODEL_FILE):
        return None
    try:
        data = joblib.load(MODEL_FILE)
        # Handle dictionary bundle or bare model
        if isinstance(data, dict):
            return data
        else:
            return {"model": data, "features": None, "classes": list(data.classes_), "normal_stats": None}
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

bundle = load_bundle()

# Header
st.markdown('<div class="main-title">⚙️ AI-Based Industrial Fault Diagnosis & Root Cause Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Multi-Class Fault Classification, Sensor Anomaly Detection & Prescriptive Maintenance</div>', unsafe_allow_html=True)

if bundle is None:
    st.error("❌ Trained model file (`model.pkl`) was not found.")
    st.info("💡 Run `python train_model.py` in your terminal to train and save the model.")
    st.stop()

model = bundle["model"]
normal_stats = bundle.get("normal_stats", {})

# Sidebar: Quick Presets & System Info
with st.sidebar:
    st.header("🎛️ Diagnostic Presets")
    st.write("Select a scenario to populate realistic sensor readings:")

    # Initialize session state defaults
    if "air_temp" not in st.session_state:
        st.session_state.air_temp = 300.0
        st.session_state.proc_temp = 310.0
        st.session_state.rot_speed = 1530
        st.session_state.torque = 40.0
        st.session_state.tool_wear = 50

    if st.button("🟢 Normal Operation Preset", use_container_width=True):
        st.session_state.air_temp = 300.0
        st.session_state.proc_temp = 310.0
        st.session_state.rot_speed = 1530
        st.session_state.torque = 40.0
        st.session_state.tool_wear = 45
        st.rerun()

    if st.button("🔥 Overheating Preset", use_container_width=True):
        st.session_state.air_temp = 304.5
        st.session_state.proc_temp = 323.0
        st.session_state.rot_speed = 1510
        st.session_state.torque = 41.5
        st.session_state.tool_wear = 70
        st.rerun()

    if st.button("⚡ Excessive Torque Preset", use_container_width=True):
        st.session_state.air_temp = 299.8
        st.session_state.proc_temp = 311.2
        st.session_state.rot_speed = 1220
        st.session_state.torque = 74.0
        st.session_state.tool_wear = 85
        st.rerun()

    if st.button("⏱️ Excessive Tool Wear Preset", use_container_width=True):
        st.session_state.air_temp = 300.5
        st.session_state.proc_temp = 312.0
        st.session_state.rot_speed = 1520
        st.session_state.torque = 48.0
        st.session_state.tool_wear = 230
        st.rerun()

    if st.button("🔄 Abnormal Speed Preset", use_container_width=True):
        st.session_state.air_temp = 300.1
        st.session_state.proc_temp = 310.5
        st.session_state.rot_speed = 2650
        st.session_state.torque = 32.0
        st.session_state.tool_wear = 60
        st.rerun()

    st.markdown("---")
    st.markdown("""
    **Architecture Overview:**
    * **Algorithm:** Random Forest Classifier
    * **Diagnostic Target:** 5 Categorical Modes
    * **Inference Method:** Multi-Class Probability
    * **Decision Layer:** Rule-Based RCA Engine
    """)

# Main Section: Sensor Inputs
st.subheader("📋 Machine Telemetry Inputs")

col1, col2 = st.columns(2)

with col1:
    air_temp = st.number_input(
        "🌡️ Air Temperature [Kelvin]",
        min_value=280.0,
        max_value=330.0,
        value=float(st.session_state.air_temp),
        step=0.1,
        help="Ambient temperature surrounding machinery (Normal: ~296K to 304K)"
    )

    proc_temp = st.number_input(
        "🔥 Process Temperature [Kelvin]",
        min_value=290.0,
        max_value=345.0,
        value=float(st.session_state.proc_temp),
        step=0.1,
        help="Operating core temperature of the equipment (Normal: ~306K to 314K)"
    )

    rot_speed = st.number_input(
        "🔄 Rotational Speed [RPM]",
        min_value=700,
        max_value=3500,
        value=int(st.session_state.rot_speed),
        step=10,
        help="Spindle rotational speed (Nominal: ~1350 to 1750 RPM)"
    )

with col2:
    torque_val = st.number_input(
        "⚡ Torque [Newton-meters (Nm)]",
        min_value=5.0,
        max_value=100.0,
        value=float(st.session_state.torque),
        step=0.5,
        help="Motor rotational twisting force (Nominal: ~25 to 55 Nm)"
    )

    tool_wear_val = st.number_input(
        "⏱️ Tool Wear [Minutes]",
        min_value=0,
        max_value=350,
        value=int(st.session_state.tool_wear),
        step=1,
        help="Cumulative cutter operation time (Nominal: < 190 mins)"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    diagnose_btn = st.button("🔍 Run Fault Diagnosis & Telemetry Analysis", type="primary", use_container_width=True)

# Helper: Sensor Anomaly Detection (Rule-based check against nominal baselines)
def check_sensor_anomalies(air_t, proc_t, speed, trq, wear):
    anomalies = []
    
    # Thermal checks
    temp_diff = proc_t - air_t
    if proc_t > 318.0 or temp_diff > 16.0:
        anomalies.append({
            "sensor": "Process Temperature",
            "value": f"{proc_t:.1f} K (ΔT: {temp_diff:.1f} K)",
            "status": "🔴 Critical",
            "observation": "High internal operating temperature / high thermal gradient."
        })
    elif proc_t > 314.0 or temp_diff > 13.0:
        anomalies.append({
            "sensor": "Process Temperature",
            "value": f"{proc_t:.1f} K (ΔT: {temp_diff:.1f} K)",
            "status": "🟡 Elevated",
            "observation": "Operating temperature near upper boundary."
        })
    else:
        anomalies.append({
            "sensor": "Process Temperature",
            "value": f"{proc_t:.1f} K (ΔT: {temp_diff:.1f} K)",
            "status": "🟢 Nominal",
            "observation": "Thermal operating conditions within normal parameters."
        })

    # Ambient Air check
    if air_t > 304.0:
        anomalies.append({
            "sensor": "Air Temperature",
            "value": f"{air_t:.1f} K",
            "status": "🟡 Elevated",
            "observation": "Factory ambient room temperature is higher than typical baseline."
        })
    else:
        anomalies.append({
            "sensor": "Air Temperature",
            "value": f"{air_t:.1f} K",
            "status": "🟢 Nominal",
            "observation": "Ambient factory air within comfortable baseline."
        })

    # Torque check
    if trq > 60.0:
        anomalies.append({
            "sensor": "Torque",
            "value": f"{trq:.1f} Nm",
            "status": "🔴 Critical",
            "observation": "Torque exceeds 60 Nm threshold indicating mechanical overstrain."
        })
    elif trq > 52.0:
        anomalies.append({
            "sensor": "Torque",
            "value": f"{trq:.1f} Nm",
            "status": "🟡 Elevated",
            "observation": "Torque slightly elevated above nominal duty range."
        })
    else:
        anomalies.append({
            "sensor": "Torque",
            "value": f"{trq:.1f} Nm",
            "status": "🟢 Nominal",
            "observation": "Mechanical drive load is balanced."
        })

    # Speed check
    if speed < 1200 or speed > 2200:
        anomalies.append({
            "sensor": "Rotational Speed",
            "value": f"{speed} RPM",
            "status": "🔴 Critical",
            "observation": "RPM sits well outside nominal operating window (1350 - 1750 RPM)."
        })
    elif speed < 1300 or speed > 1850:
        anomalies.append({
            "sensor": "Rotational Speed",
            "value": f"{speed} RPM",
            "status": "🟡 Elevated",
            "observation": "Spindle speed slightly outside optimal operating envelope."
        })
    else:
        anomalies.append({
            "sensor": "Rotational Speed",
            "value": f"{speed} RPM",
            "status": "🟢 Nominal",
            "observation": "Spindle speed is stable."
        })

    # Tool wear check
    if wear > 200:
        anomalies.append({
            "sensor": "Tool Wear",
            "value": f"{wear} mins",
            "status": "🔴 Critical",
            "observation": "Cutter wear exceeds maximum recommended service life (200 min)."
        })
    elif wear > 160:
        anomalies.append({
            "sensor": "Tool Wear",
            "value": f"{wear} mins",
            "status": "🟡 Elevated",
            "observation": "Cutter wear approaching end-of-life replacement threshold."
        })
    else:
        anomalies.append({
            "sensor": "Tool Wear",
            "value": f"{wear} mins",
            "status": "🟢 Nominal",
            "observation": "Cutter wear is within safe operational limits."
        })

    return anomalies

# Execution and Results
if diagnose_btn:
    input_df = pd.DataFrame([{
        "air_temperature": air_temp,
        "process_temperature": proc_temp,
        "rotational_speed": rot_speed,
        "torque": torque_val,
        "tool_wear": tool_wear_val
    }])

    # 1. Statistical ML Model Prediction
    predicted_category = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    classes = list(model.classes_)
    
    # Format probabilities into dictionary
    prob_dict = {cls: prob * 100 for cls, prob in zip(classes, probabilities)}
    confidence = prob_dict[predicted_category]

    st.markdown("---")
    
    # 2. Diagnosis Display
    st.subheader("🎯 Diagnostic Classification Result")
    
    if predicted_category == "Normal Operation":
        st.markdown(f"""
        <div class="card-normal">
            <h3>🟢 {predicted_category}</h3>
            <p>The statistical model classifies current machine telemetry as healthy with <strong>{confidence:.1f}% confidence</strong>. All sensor readings are operating within normal operational tolerances.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="card-fault">
            <h3>⚠️ Fault Detected: {predicted_category}</h3>
            <p>The machine learning classifier identified abnormal telemetry patterns matching <strong>{predicted_category}</strong> with <strong>{confidence:.1f}% model confidence</strong>.</p>
        </div>
        """, unsafe_allow_html=True)

    # Probability Distribution Breakdown
    st.write("")
    with st.expander("📊 View Model Probability Distribution Across All Fault Modes", expanded=False):
        prob_df = pd.DataFrame({
            "Fault Category": list(prob_dict.keys()),
            "Model Confidence (%)": [round(v, 2) for v in prob_dict.values()]
        }).sort_values(by="Model Confidence (%)", ascending=False)
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

    # 3. Sensor Anomaly Detection Table
    st.markdown("---")
    st.subheader("📡 Sensor Anomaly Detection")
    st.caption("Individual sensor validation against historical healthy operating boundaries:")
    
    sensor_checks = check_sensor_anomalies(air_temp, proc_temp, rot_speed, torque_val, tool_wear_val)
    anom_df = pd.DataFrame(sensor_checks)
    st.dataframe(anom_df, use_container_width=True, hide_index=True)

    # 4. Root Cause Analysis (Rule-Based Explanations)
    st.markdown("---")
    st.subheader("🔬 Root Cause Analysis (Potential Contributing Factors)")
    st.caption("⚠️ Note: These are rule-based engineering explanations of sensor patterns, not guaranteed physical confirmations.")

    rca_points = []
    maintenance_actions = []

    if predicted_category == "Overheating":
        rca_points.append("Process temperature or process-to-ambient temperature delta is running high.")
        rca_points.append("Possible physical causes: Inadequate cooling airflow, blocked ventilation channels, or degraded thermal dissipation.")
        maintenance_actions.append("Inspect machine cooling system and coolant reservoir levels.")
        maintenance_actions.append("Check and clean ventilation filters and heat exchanger fins.")
        maintenance_actions.append("Verify ambient room temperature and air conditioning circulation around the cabinet.")

    elif predicted_category == "Excessive Torque":
        rca_points.append("Measured torque exceeds the normal operating threshold (> 60 Nm).")
        rca_points.append("Possible physical causes: Mechanical binding, improper spindle lubrication, material feed resistance, or motor drive overstrain.")
        maintenance_actions.append("Halt spindle and inspect drive shaft for mechanical resistance or physical jamming.")
        maintenance_actions.append("Check lubricant levels and inspect drive belt / gear coupling condition.")
        maintenance_actions.append("Recalibrate feed rate to ensure cutting material does not exceed allowable chip load.")

    elif predicted_category == "Excessive Tool Wear":
        rca_points.append("Cumulative cutter usage exceeds 200 minutes under continuous duty.")
        rca_points.append("Possible physical causes: Normal abrasive wear of the cutting insert leading to dull edges and increased cutting friction.")
        maintenance_actions.append("Schedule immediate tool bit inspection and replace cutting insert.")
        maintenance_actions.append("Inspect machined workpiece surface roughness for chatter marks or dimensional drift.")
        maintenance_actions.append("Reset tool wear timer following insert replacement.")

    elif predicted_category == "Abnormal Rotational Speed":
        rca_points.append("Spindle RPM significantly deviates from the nominal operating window (1350 - 1750 RPM).")
        rca_points.append("Possible physical causes: Variable Frequency Drive (VFD) governor instability, belt slip, bearing friction, or motor encoder fault.")
        maintenance_actions.append("Check motor speed encoder signal and VFD frequency calibration.")
        maintenance_actions.append("Inspect spindle bearings for acoustic vibration or radial runout.")
        maintenance_actions.append("Verify power supply voltage stability to the motor controller.")

    else:
        rca_points.append("All sensor readings align with standard baseline operating parameters.")
        rca_points.append("No adverse physical mechanisms or anomalous trends observed in current telemetry.")
        maintenance_actions.append("Continue regular operating shift.")
        maintenance_actions.append("Maintain routine visual inspections and scheduled preventive maintenance rounds.")

    st.markdown('<div class="rca-box"><h4>🔍 Potential Telemetry Explanation</h4><ul>' + 
                "".join([f"<li>{p}</li>" for p in rca_points]) + "</ul></div>", unsafe_allow_html=True)

    # 5. Maintenance Recommendations
    st.markdown('<div class="maintenance-box"><h4>🛠️ Recommended Maintenance Checklist</h4><ul>' + 
                "".join([f"<li>{a}</li>" for a in maintenance_actions]) + "</ul></div>", unsafe_allow_html=True)

# Footer & Disclaimer
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Educational Demonstration Notice & Disclaimer:</strong><br>
    This project is an educational demonstration built using <strong>synthetic demonstration telemetry</strong>.
    Model predictions are statistical estimations and rule-based explanations represent <em>possible contributing factors</em> based on sensor thresholds.
    This software is <strong>not a certified industrial safety system</strong> and should never be used as the sole safety mechanism on real industrial equipment.
</div>
""", unsafe_allow_html=True)
