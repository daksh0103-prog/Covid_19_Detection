import streamlit as st
import pandas as pd
import pickle

def load_model():
    with open("covid_disease_detection.pkl", "rb") as file:
        model = pickle.load(file)
    return model


model = load_model()

st.set_page_config(page_title="Covid-19 Detection", layout="wide")

st.image('https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExZWdmeGN5YTloOGY4c2k4dTNrNjJnYWxjenI1OHdkOXJ0MzBqNTUyNSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/RPqLAs6u4z0YUTsOCM/giphy.gif')
st.title("COVID Risk Prediction")

st.write("""
COVID-19 has seen global fluctuations, with a significant peak and subsequent decline but not a complete disappearance;
"prediction" in a scientific context often refers to modeling future trends, not a singular forecast. Predicting the exact behavior of the virus is 
difficult due to factors like new variants, changing human behavior, and the development of immunity.
""")


# List of all possible features (the columns your model was trained on)
all_possible_columns = [
    "Breathing Problem", "Fever", "Dry Cough", "Sore throat", "Running Nose",
    "Asthma", "Chronic Lung Disease", "Headache", "Heart Disease", "Diabetes",
    "Hyper Tension", "Fatigue ", "Gastrointestinal ", "Abroad travel",
    "Contact with COVID Patient", "Attended Large Gathering",
    "Visited Public Exposed Places", "Family working in Public Exposed Places"
]



# User selects symptoms from a multiselect
user_input = st.multiselect(
    "Select Symptoms or Risk Factors you are experiencing:",
    options=all_possible_columns
)

# Prepare input for the model: map to 0/1
cleaned_input = {col: 1 if col in user_input else 0 for col in all_possible_columns}

# Convert to DataFrame
input_df = pd.DataFrame([cleaned_input])

# Show input to user for confirmation (optional)
st.write("Your input data:")
st.dataframe(input_df)

# Prediction Button
if st.button("Predict COVID Risk"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("⚠ High Risk of COVID. Please consult a doctor immediately!")
    else:
        st.success("✅ Low Risk of COVID. Continue monitoring your health.")