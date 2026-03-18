from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('optimized_xgboost.joblib')
scaler = joblib.load('scaler.joblib')

expected_columns = [
    'gender', 'region', 'highest_education', 'imd_band', 'age_band',
    'num_of_prev_attempts', 'studied_credits', 'score'
]

features_to_scale = ['num_of_prev_attempts', 'studied_credits', 'score']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Validate input
        for col in expected_columns:
            if col not in data:
                return jsonify({'status': 'error', 'message': f'Missing field: {col}'})

        input_df = pd.DataFrame([data])
        input_df = input_df[expected_columns]

        # Convert numeric
        input_df[features_to_scale] = input_df[features_to_scale].astype(float)

        # Scale
        input_df[features_to_scale] = scaler.transform(input_df[features_to_scale])

        prediction = model.predict(input_df)

        return jsonify({
            'status': 'success',
            'prediction': int(prediction[0])
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
