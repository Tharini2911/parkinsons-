from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("parkinson_model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    try:
        features = []

        for value in request.form.values():
            features.append(float(value))

        # Convert to NumPy array
        features = np.array(features).reshape(1, -1)

        # Scale the input
        features = scaler.transform(features)

        # Predict
        prediction = model.predict(features)

        return render_template(
            "index.html",
            prediction_text=f"Predicted total_UPDRS : {prediction[0]:.2f}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error : {e}"
        )


if __name__ == "__main__":
    app.run(debug=True)