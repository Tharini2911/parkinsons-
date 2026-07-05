import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("parkinson_model.pkl")

st.title("Parkinson's Disease Progression Prediction")

subject = st.text_input("Subject Number")
age = st.text_input("Age")
sex = st.text_input("Sex (0=Female, 1=Male)")
test_time = st.text_input("Test Time")
motor_UPDRS = st.text_input("Motor UPDRS")

jitter_percent = st.text_input("Jitter (%)")
jitter_abs = st.text_input("Jitter (Abs)")
jitter_rap = st.text_input("Jitter RAP")
jitter_ppq5 = st.text_input("Jitter PPQ5")
jitter_ddp = st.text_input("Jitter DDP")

shimmer = st.text_input("Shimmer")
shimmer_db = st.text_input("Shimmer dB")
shimmer_apq3 = st.text_input("Shimmer APQ3")
shimmer_apq5 = st.text_input("Shimmer APQ5")
shimmer_apq11 = st.text_input("Shimmer APQ11")
shimmer_dda = st.text_input("Shimmer DDA")

nhr = st.text_input("NHR")
hnr = st.text_input("HNR")
rpde = st.text_input("RPDE")
dfa = st.text_input("DFA")
ppe = st.text_input("PPE")

if st.button("Predict"):
    try:
        input_data = pd.DataFrame([[
            float(subject), float(age), float(sex),
            float(test_time), float(motor_UPDRS),
            float(jitter_percent), float(jitter_abs),
            float(jitter_rap), float(jitter_ppq5),
            float(jitter_ddp), float(shimmer),
            float(shimmer_db), float(shimmer_apq3),
            float(shimmer_apq5), float(shimmer_apq11),
            float(shimmer_dda), float(nhr),
            float(hnr), float(rpde),
            float(dfa), float(ppe)
        ]], columns=[
            'subject#', 'age', 'sex', 'test_time', 'motor_UPDRS',
            'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP',
            'Jitter:PPQ5', 'Jitter:DDP',
            'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3',
            'Shimmer:APQ5', 'Shimmer:APQ11',
            'Shimmer:DDA', 'NHR', 'HNR',
            'RPDE', 'DFA', 'PPE'
        ])

        prediction = model.predict(input_data)
        st.success(f"Predicted total_UPDRS : {prediction[0]:.2f}")

    except ValueError:
        st.error("Please enter valid numeric values in all fields.")