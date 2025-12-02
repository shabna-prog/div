import streamlit as st
import pandas as pd
import joblib

# Load the trained model
filename = r'Logistic_Model.sav'
loaded_model = joblib.load(open(filename, 'rb'))

# Correct column names (must match training EXACTLY)
columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

# Prediction function
def Diabetes_prediction(features):
    prediction = loaded_model.predict(features)
    return prediction

# Streamlit UI
st.title("Diabetes Prediction")

st.write("Please provide the following information:")

Pregnancies = st.number_input("Pregnancies", min_value=0)
Glucose = st.number_input("Glucose", min_value=0)
BloodPressure = st.number_input("BloodPressure", min_value=0)
SkinThickness = st.number_input("SkinThickness", min_value=0)
Insulin = st.number_input("Insulin", min_value=0)
BMI = st.number_input("BMI", min_value=0.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0)
Age = st.number_input("Age", min_value=0)

# Create dataframe
input_data = pd.DataFrame(
    [[Pregnancies, Glucose, BloodPressure, SkinThickness,
      Insulin, BMI, DiabetesPedigreeFunction, Age]],
    columns=columns
)

# Debug info
st.write("Model expects:", loaded_model.n_features_in_)
st.write("You passed:", input_data.shape[1])

# Prediction
if st.button("Diabetes Prediction"):
    prediction = Diabetes_prediction(input_data)
    if prediction[0] == 0:
        st.write("Predicted Diabetes: 0 (No diabetes)")
    else:
        st.write("Predicted Diabetes: 1 (Diabetes predicted)")

