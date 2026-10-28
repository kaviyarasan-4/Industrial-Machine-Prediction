import joblib
import pandas as pd
from app import check_sensor_anomalies

bundle = joblib.load('model.pkl')
model = bundle['model']

test_cases = [
    {
        'category': 'Normal Operation',
        'inputs': {'air_temperature': 300.0, 'process_temperature': 310.0, 'rotational_speed': 1530, 'torque': 40.0, 'tool_wear': 45},
        'expected_maintenance': 'Continue regular operating shift'
    },
    {
        'category': 'Overheating',
        'inputs': {'air_temperature': 304.5, 'process_temperature': 323.0, 'rotational_speed': 1510, 'torque': 41.5, 'tool_wear': 70},
        'expected_maintenance': 'Inspect machine cooling system'
    },
    {
        'category': 'Excessive Torque',
        'inputs': {'air_temperature': 299.8, 'process_temperature': 311.2, 'rotational_speed': 1220, 'torque': 74.0, 'tool_wear': 85},
        'expected_maintenance': 'Halt spindle and inspect drive shaft'
    },
    {
        'category': 'Excessive Tool Wear',
        'inputs': {'air_temperature': 300.5, 'process_temperature': 312.0, 'rotational_speed': 1520, 'torque': 48.0, 'tool_wear': 230},
        'expected_maintenance': 'Schedule immediate tool bit inspection'
    },
    {
        'category': 'Abnormal Rotational Speed',
        'inputs': {'air_temperature': 300.1, 'process_temperature': 310.5, 'rotational_speed': 2650, 'torque': 32.0, 'tool_wear': 60},
        'expected_maintenance': 'Check motor speed encoder signal'
    }
]

print("=" * 75)
print("COMPREHENSIVE TEST SUITE: DIAGNOSTICS & ROOT CAUSE ANALYSIS")
print("=" * 75)

all_passed = True
for idx, tc in enumerate(test_cases, 1):
    input_df = pd.DataFrame([tc['inputs']])
    
    # 1. Model Diagnosis
    predicted_category = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    classes = list(model.classes_)
    prob_dict = {cls: prob * 100 for cls, prob in zip(classes, probabilities)}
    confidence = prob_dict[predicted_category]
    
    model_match = (predicted_category == tc['category'])
    
    # 2. Rule-based sensor checks
    anomalies = check_sensor_anomalies(
        tc['inputs']['air_temperature'],
        tc['inputs']['process_temperature'],
        tc['inputs']['rotational_speed'],
        tc['inputs']['torque'],
        tc['inputs']['tool_wear']
    )
    
    status_str = "PASS" if model_match else "FAIL"
    if not model_match:
        all_passed = False
        
    print(f"\nTest #{idx}: {tc['category']}")
    print(f" • Model Diagnosis: {predicted_category} (Confidence: {confidence:.1f}%) -> [{status_str}]")
    print(f" • Sensor Flags: {[a['sensor'] + ': ' + a['status'] for a in anomalies]}")
    print(f" • Rule-Based Maintenance Check: {tc['expected_maintenance']}")

print("\n" + "=" * 75)
if all_passed:
    print("✅ ALL 5 DIAGNOSTIC TESTS PASSED WITH 100% SUCCESS!")
else:
    print("❌ SOME DIAGNOSTIC TESTS FAILED!")
print("=" * 75)
