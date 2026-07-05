import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("parkinson_model.pkl")

st.title("Parkinson's Disease Progression Prediction")

subject = st.number_input("Subject Number", value=1)
age = st.number_input("Age", value=60)
sex = st.number_input("Sex (0=Female, 1=Male)", value=1)
test_time = st.number_input("Test Time", value=1.0)
motor_UPDRS = st.number_input("Motor UPDRS", value=20.0)

jitter_percent = st.number_input("Jitter (%)", value=0.01)
jitter_abs = st.number_input("Jitter (Abs)", value=0.00005)
jitter_rap = st.number_input("Jitter RAP", value=0.005)
jitter_ppq5 = st.number_input("Jitter PPQ5", value=0.005)
jitter_ddp = st.number_input("Jitter DDP", value=0.015)

shimmer = st.number_input("Shimmer", value=0.03)
shimmer_db = st.number_input("Shimmer dB", value=0.3)
shimmer_apq3 = st.number_input("Shimmer APQ3", value=0.02)
shimmer_apq5 = st.number_input("Shimmer APQ5", value=0.03)
shimmer_apq11 = st.number_input("Shimmer APQ11", value=0.04)
shimmer_dda = st.number_input("Shimmer DDA", value=0.06)

nhr = st.number_input("NHR", value=0.02)
hnr = st.number_input("HNR", value=20.0)
rpde = st.number_input("RPDE", value=0.5)
dfa = st.number_input("DFA", value=0.7)
ppe = st.number_input("PPE", value=0.2)

if st.button("Predict"):

    input_data = pd.DataFrame([[
        subject, age, sex, test_time, motor_UPDRS,
        jitter_percent, jitter_abs, jitter_rap,
        jitter_ppq5, jitter_ddp,
        shimmer, shimmer_db, shimmer_apq3,
        shimmer_apq5, shimmer_apq11,
        shimmer_dda, nhr, hnr,
        rpde, dfa, ppe
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