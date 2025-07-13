import pandas as pd
import requests
import json

# Test the enhanced profiling functionality
def test_enhanced_profiling():
    # Create test medical data
    test_data = {
        'patient_id': range(1, 11),
        'age': [25, 34, 45, 67, 23, 56, 78, 32, 41, 29],
        'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
        'weight': [70.5, 62.3, 85.2, 68.7, 75.1, 58.9, 82.4, 71.8, 79.2, 66.5],
        'height': [175, 162, 180, 158, 178, 165, 172, 169, 176, 160],
        'blood_pressure_systolic': [120, 135, 140, 150, 118, 145, 160, 125, 138, 122],
        'glucose': [95, 110, 125, 140, 88, 120, 155, 98, 115, 92]
    }
    
    df = pd.DataFrame(test_data)
    df.to_csv('/tmp/test_medical_data.csv', index=False)
    
    print("✅ Test medical data created successfully")
    print(f"📊 Dataset: {len(df)} patients × {len(df.columns)} variables")
    print(f"🏥 Medical variables detected: age, gender, weight, height, blood_pressure, glucose")
    
    return '/tmp/test_medical_data.csv'

if __name__ == "__main__":
    csv_path = test_enhanced_profiling()
    print(f"\n🎯 Test data ready at: {csv_path}")
    print("\n💡 Upload this file through the frontend to test enhanced profiling!")