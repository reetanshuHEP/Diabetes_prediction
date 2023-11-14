# Import necessary libraries
import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))


# creating a function for Prediction

def diabetes_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)  # Convert input to float

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'


# ... (your existing code)

def main():
    st.title('Diabetes Prediction Web App')

    # Add a brief description
    st.markdown("This app predicts whether a person has diabetes based on input features.")

    # Creating a sidebar for the input data
    st.sidebar.header('Enter the following values')

    # Group related input fields
    st.sidebar.subheader('Pregnancy Information')
    Pregnancies = st.sidebar.number_input('Number of Pregnancies', min_value=0)

    st.sidebar.subheader('Glucose and Blood Pressure')
    Glucose = st.sidebar.number_input('Glucose Level (mg/dL)', min_value=0)
    BloodPressure = st.sidebar.number_input('Blood Pressure (mm Hg)', min_value=0)

    st.sidebar.subheader('Skin Thickness, Insulin, and BMI')
    SkinThickness = st.sidebar.number_input('Skin Thickness (mm)', min_value=0)
    Insulin = st.sidebar.number_input('Insulin Level (mu U/mL)', min_value=0)
    BMI = st.sidebar.number_input('BMI', min_value=0.0)

    st.sidebar.subheader('Other Information')
    DiabetesPedigreeFunction = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0)
    Age = st.sidebar.number_input('Age of the Person', min_value=0)

    # Code for Prediction
    diagnosis = ''

    # Creating a button for Prediction
    if st.button('Predict'):
        diagnosis = diabetes_prediction(
            [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    # Displaying the result
    st.subheader('The result is:')
    st.success(diagnosis)

if __name__ == '__main__':
    main()
