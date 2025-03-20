from flask import Flask, render_template, request
import numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Load the trained model, polynomial transformer, and scaler
model = pickle.load(open('model.pkl', 'rb'))
poly = pickle.load(open('poly.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))  # Load the scaler

app = Flask(__name__)

@app.route('/', methods=['GET'])
def land():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def prediction():
    try:
        # Debugging: Print received data
        print("Received form data:", request.form)

        # Convert form data to float
        age = float(request.form['age'])
        bmi = float(request.form['bmi'])
        HbA1c_level = float(request.form['HbA1c_level'])
        blood_glucose_level = float(request.form['blood_glucose_level'])
        hypertension = 1 if request.form['hypertension'] == '1' else 0
        heart_disease = 1 if request.form['heart_disease'] == '1' else 0

        # Log transform age
        age_log = np.log1p(age)

        # Normalize numerical features using the same scaler
        scaled_features = scaler.transform([[bmi, HbA1c_level, blood_glucose_level]])

        # Apply polynomial transformation
        X_poly = poly.transform(scaled_features)

        # Combine all features
        new_inputs = np.hstack((np.array([age_log, hypertension, heart_disease]), X_poly.flatten())).reshape(1, -1)

        # Debugging: Print processed input
        print("Processed input:", new_inputs)

        # Predict probability
        prediction_prob = model.predict_proba(new_inputs)[0][1]

        # Apply threshold (0.35)
        y_pred_done = int(prediction_prob >= 0.35)

        # Prediction message
        prediction_text = f"The model predicts: {'Diabetes' if y_pred_done == 1 else 'No Diabetes'} (Chances of Diabetes: {prediction_prob:.2f}%)"

        return render_template('index.html', prediction_text=prediction_text)

    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html', prediction_text="Error with prediction.")

if __name__ == '__main__':
    app.run(port=3000, debug=True)