
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
filename = r'Logistic_Model.sav'
loaded_model = joblib.load(open(filename, 'rb'))

# Define the correct column names
columns = ['Pregnacies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'DiabetesPedigreeFunction',
           'Age']

# Define the prediction function
def Diabetes_prediction(features):
    """
    Predicts the diabetes prediciton based on input features.
    """
    prediction = loaded_model.predict(features)
    return prediction

# Create the Streamlit app
st.title("Diabetes Prediction")

# Get user input
st.write("Please provide the following information:")
Pregnancies = st.number_input("Pregnancies", min_value=0.0)
Glucose = st.number_input("Glucose", min_value=1, max_value=5)
BloodPressure = st.number_input("BloodPressure", min_value=1, max_value=5)
Insulin = st.number_input("Insulin", min_value=1)
BMI = st.number_input("BMI", min_value=0.0)
DiabetesPredictionFunction= st.number_input("DiabetesPredictionFunction", min_value=0)
Age = st.number_input("Age", min_value=0.0)

# Create a dataframe with the user input
input_data = pd.DataFrame([[Pregnancies,Glucose,BloodPressure,Insulin,BMI,DiabetesPredictionFunction,Age]], columns=columns)

# Make a prediction
# Make a prediction
if st.button("Diabetes Prediction"):
    prediction = Diabetes_prediction(input_data)
    if prediction[0] == 0:
        st.write("Predicted Diabetes: 0 (No significant delay expected)")
    else:
        st.write("Predicted Diabetes: 1 (Delay expected)")
             
