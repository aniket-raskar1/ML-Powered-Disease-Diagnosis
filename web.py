import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from pathlib import Path

st.set_page_config(page_title="Prediction of Diseases Outbreak",
                   layout='wide',
                   page_icon='🧑‍⚕️')


MODEL_DIR = Path("Trained Models")

# Load models
diabetes_model = pickle.load(open(MODEL_DIR / "diabetes_model.pkl", "rb"))
heart_model = pickle.load(open(MODEL_DIR / "heart_model.pkl", "rb"))
parkinsons_model = pickle.load(open(MODEL_DIR / "parkinsons_model.pkl", "rb"))

with st.sidebar:
    selected = option_menu('Prediction of Diseases Outbreak System',
                           ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinson’s Disease Prediction'],
                           menu_icon='hospital-fill', icons=['activity', 'heart', 'person'], default_index=0)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.number_input('No of Pregnancies:', min_value=0, step=1)
    with col2:
        Glucose = st.number_input('Glucose Level:', min_value=0)
    with col3:
        Bloodpressure = st.number_input("Blood Pressure Level:", min_value=0)
    with col1:
        SkinThickness = st.number_input('Skin Thickness Value:', min_value=0)
    with col2:
        Insulin = st.number_input("Insulin Level:", min_value=0)
    with col3:
        BMI = st.number_input("BMI Value:", min_value=0.0, format="%.2f")
    with col1:
        DiabetesPredigreefunction = st.number_input('Diabetes Pedigree Function:', min_value=0.0, format="%.3f")
    with col2:
        Age = st.number_input('Age:', min_value=0, step=1)

    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, Bloodpressure, SkinThickness, Insulin, BMI, DiabetesPredigreefunction, Age]
        diab_prediction = diabetes_model.predict([user_input])
        if all(x == 0 for x in user_input):
            st.warning("All fields are required and must be greater than zero except Pregnencies")
        if diab_prediction[0] == 1:
            st.success('The Person is Diabetic')
        else:
            st.success('The Person is Not Diabetic')


if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')
    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.number_input('Age:', min_value=0, step=1)
        Sex = st.selectbox('Sex:', ['Male', 'Female'])
        Sex = 1 if Sex == 'Male' else 0
        ChestPainType = st.selectbox('Chest Pain Type:', [0, 1, 2, 3])
        RestECG = st.selectbox('Resting ECG:', [0, 1, 2]) 
    with col2:
        RestingBP = st.number_input('Resting Blood Pressure:', min_value=0)
        Cholesterol = st.number_input('Cholesterol Level:', min_value=0)
        FastingBS = st.selectbox('Fasting Blood Sugar > 120 mg/dL:', [0, 1])
        Oldpeak = st.number_input('Oldpeak:', min_value=0.0, format="%.1f")  

    with col3:
        MaxHR = st.number_input('Maximum Heart Rate Achieved:', min_value=0)
        ExerciseAngina = st.selectbox('Exercise Induced Angina:', [0, 1])
        ST_Slope = st.selectbox('ST Slope:', [0, 1, 2])
        CA = st.number_input('Number of Major Vessels (0-3):', min_value=0, max_value=3, step=1) 
        Thal = st.selectbox('Thalassemia:', [0, 1, 2, 3])  

    if st.button('Heart Disease Test Result'):
        user_input = [Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, CA, Thal]
        
        heart_prediction = heart_model.predict([user_input])
        if heart_prediction[0] == 1:
            st.success('The Person haslsls Heart Disease')
        else:
            st.success('The Person does not have Heart Disease')


# Parkinson’s Disease Prediction Page
if selected == "Parkinson’s Disease Prediction":
    st.title("Parkinson’s Disease Prediction using ML")
    col1, col2, col3 = st.columns(3)

    with col1:
        MDVP_Fo = st.number_input('MDVP:Fo(Hz)')
        MDVP_Fhi = st.number_input('MDVP:Fhi(Hz)')
        MDVP_Flo = st.number_input('MDVP:Flo(Hz)')
        MDVP_Jitter = st.number_input('MDVP:Jitter(%)')
        MDVP_Jitter_Abs = st.number_input('MDVP:Jitter(Abs)')
        MDVP_RAP = st.number_input('MDVP:RAP')
        MDVP_PPQ = st.number_input('MDVP:PPQ')
        Jitter_DDP = st.number_input('Jitter:DDP')
        D2 = st.number_input('D2')

    with col2:
        MDVP_Shimmer = st.number_input('MDVP:Shimmer')
        MDVP_Shimmer_dB = st.number_input('MDVP:Shimmer(dB)')
        Shimmer_APQ3 = st.number_input('Shimmer:APQ3')
        Shimmer_APQ5 = st.number_input('Shimmer:APQ5')
        MDVP_APQ = st.number_input('MDVP:APQ')
        Shimmer_DDA = st.number_input('Shimmer:DDA')
        NHR = st.number_input('NHR')
        HNR = st.number_input('HNR')
        DFA = st.number_input('DFA')

    with col3:
        RPDE = st.number_input('RPDE')
       
        Spread1 = st.number_input('Spread1')
        Spread2 = st.number_input('Spread2')
        PPE = st.number_input('PPE')
        

    if st.button("Parkinson’s Test Result"):
        user_input = [MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitter, MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ, Jitter_DDP,
                      MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR,
                      RPDE, DFA, Spread1, Spread2,D2, PPE]

        if len(user_input) == parkinson_model.n_features_in_:
            parkinson_prediction = parkinson_model.predict([user_input])
            if parkinson_prediction[0] == 1:
                st.success('The Person has Parkinson’s Disease')
            else:
                st.success('The Person does not have Parkinson’s Disease')
        else:
            st.error(f"Expected {parkinson_model.n_features_in_} features, but got {len(user_input)}. Check inputs.")

