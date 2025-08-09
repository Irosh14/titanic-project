# app.py - simple Streamlit app to make predictions
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Titanic Predictor")

st.title("Titanic Survival Predictor ")

# load model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.write("Enter passenger details and click Predict.")

pclass = st.selectbox("Pclass", [1,2,3], index=2)
sex = st.selectbox("Sex", ["male","female"])
age = st.number_input("Age", min_value=0.0, max_value=120.0, value=30.0)
sibsp = st.number_input("SibSp (siblings/spouse)", min_value=0, max_value=10, value=0)
parch = st.number_input("Parch (parents/children)", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=32.0)
embarked = st.selectbox("Embarked", ["S","C","Q"], index=0)

if st.button("Predict"):
    X_new = pd.DataFrame([{
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked
    }])
    try:
        proba = model.predict_proba(X_new)[0][1]
        pred = model.predict(X_new)[0]
        st.success(f"Prediction: {'Survived' if pred==1 else 'Not survived'} (prob: {proba:.2f})")
    except Exception as e:
        st.error("Prediction failed. Model or inputs may be wrong.")
        st.write(e)
