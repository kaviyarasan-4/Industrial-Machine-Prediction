"""
Streamlit Web Dashboard for Industrial Machine Failure Prediction
-----------------------------------------------------------------
NOTE: This web application is an educational demonstration built on
synthetic sample data. It is NOT a certified industrial safety system.
"""

import os
import joblib
import pandas as pd
import streamlit as st

# Configure page layout and title
st.set_page_config(
    page_title="Industrial Machine Failure Prediction",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for clean, professional look
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .status-card-normal {
        background-color: #ECFDF5;
        border: 2px solid #10B981;
        border-radius: 10px;
        padding: 1.5rem;
        color: #065F46;
    }
    .status-card-failure {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 10px;
        padding: 1.5rem;
        color: #991B1B;
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

# 1. Load the trained model
MODEL_FILE = "model.pkl"

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_FILE):
        return None
    try:
        return joblib.load(MODEL_FILE)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_trained_model()

# Header Section
st.markdown('<div class="main-title">⚙️ AI-Based Industrial Machine Failure Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time predictive maintenance monitoring powered by Machine Learning</div>', unsafe_allow_html=True)

if model is None:
    st.error("❌ Trained model file (`model.pkl`) was not found in the project directory.")
    st.info("💡 To train and generate `model.pkl`, please run: `python train_model.py` in your terminal.")
    st.stop()

# Sidebar: Quick Presets & Project Info
with st.sidebar:
    st.header("🎛️ Control Panel")
    st.write("Use this dashboard to input machine sensor readings and evaluate the failure risk.")
    
    st.markdown("---")
    st.subheader("💡 Quick Test Presets")
    col_pre1, col_pre2 = st.columns(2)
    
    # Session state for sensor defaults
    if "air_temp" not in st.session_state:
        st.session_state.air_temp = 300.0
        st.session_state.proc_temp = 310.0
        st.session_state.rot_speed = 1530
        st.session_state.torque = 40.0
        st.session_state.tool_wear = 50

    if col_pre1.button("🟢 Normal"):
        st.session_state.air_temp = 300.0
        st.session_state.proc_temp = 310.0
        st.session_state.rot_speed = 1550
        st.session_state.torque = 38.0
        st.session_state.tool_wear = 45
        st.rerun()

    if col_pre2.button("🔴 High Risk"):
        st.session_state.air_temp = 304.5
        st.session_state.proc_temp = 311.0
        st.session_state.rot_speed = 1250
        st.session_state.torque = 68.0
        st.session_state.tool_wear = 220
        st.rerun()

    st.markdown("---")
    st.markdown("""
    **Model Features:**
    * Algorithm: **Random Forest**
    * Target: **Binary Classification**
    * Input Dimensions: **5 Sensors**
    """)

# Main Section: Sensor Inputs
st.subheader("📋 Enter Current Machine Sensor Readings")

col1, col2 = st.columns(2)

with col1:
    air_temp = st.number_input(
        "🌡️ Air Temperature [Kelvin]",
        min_value=280.0,
        max_value=330.0,
        value=float(st.session_state.air_temp),
        step=0.1,
        help="Ambient factory temperature around the equipment (typical: ~295K to 304K)"
    )

    proc_temp = st.number_input(
        "🔥 Process Temperature [Kelvin]",
        min_value=290.0,
        max_value=340.0,
        value=float(st.session_state.proc_temp),
        step=0.1,
        help="Internal temperature generated during machine operation (typical: ~305K to 315K)"
    )

    rot_speed = st.number_input(
        "🔄 Rotational Speed [RPM]",
        min_value=800,
        max_value=3500,
        value=int(st.session_state.rot_speed),
        step=10,
        help="Motor spindle rotation speed in revolutions per minute (typical: ~1200 to 2800 RPM)"
    )

with col2:
    torque_val = st.number_input(
        "⚡ Torque [Newton-meters (Nm)]",
        min_value=5.0,
        max_value=100.0,
        value=float(st.session_state.torque),
        step=0.5,
        help="Mechanical twisting force applied by the motor (typical: ~15 to 70 Nm)"
    )

    tool_wear_val = st.number_input(
        "⏱️ Tool Wear [Minutes]",
        min_value=0,
        max_value=350,
        value=int(st.session_state.tool_wear),
        step=1,
        help="Total operational time cutting or processing with the current tool (typical: 0 to 250 mins)"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Predict Machine Status", type="primary", use_container_width=True)

# Prediction Logic
if predict_btn:
    # 7. Exact feature names and ordering matching train_model.py
    input_data = pd.DataFrame([{
        "air_temperature": air_temp,
        "process_temperature": proc_temp,
        "rotational_speed": rot_speed,
        "torque": torque_val,
        "tool_wear": tool_wear_val
    }])

    # Generate prediction and probabilities
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    normal_prob = probabilities[0] * 100
    failure_prob = probabilities[1] * 100

    st.markdown("---")
    st.subheader("📊 Diagnostic Result")

    res_col1, res_col2 = st.columns([1.5, 1])

    with res_col1:
        if prediction == 0:
            st.markdown(f"""
            <div class="status-card-normal">
                <h2>🟢 Normal Operation</h2>
                <p>Sensor parameters are within stable operational thresholds. The machine is functioning properly with low probability of failure.</p>
                <strong>Confidence:</strong> {normal_prob:.1f}% normal
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-card-failure">
                <h2>🔴 Potential Machine Failure Risk Detected</h2>
                <p>Telemetry indicates dangerous mechanical stress, thermal imbalance, or excessive tool wear. Preventive maintenance is strongly advised.</p>
                <strong>Failure Probability:</strong> {failure_prob:.1f}% risk
            </div>
            """, unsafe_allow_html=True)

    with res_col2:
        st.write("**Failure Risk Level:**")
        st.progress(float(failure_prob / 100.0))
        st.metric(label="Calculated Risk", value=f"{failure_prob:.1f}%", delta=f"{failure_prob - 50:.1f}% vs threshold")
        
        if prediction == 1:
            st.warning("⚠️ Recommended Action: Halt spindle, inspect tool bit wear, and verify cooling.")
        else:
            st.success("✅ Machine safe to continue standard operation.")

# 9. Important synthetic demonstration disclaimer
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Educational Demonstration Notice:</strong><br>
    This project uses a synthetic demonstration dataset created for learning Python and Machine Learning concepts.
    It is <strong>not a certified industrial safety system</strong> and must not be used for real-life machinery safety operations.
</div>
""", unsafe_allow_html=True)
